#!/usr/bin/env python
# -*- coding:utf-8 -*-

from utils import inout
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa import arima_model
from dateutil.relativedelta import relativedelta


def diff_ts(ts, d):
    global shift_ts_list
    #  动态预测第二日的值时所需要的差分序列
    global last_data_shift_list
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        # print last_data_shift_list
        shift_ts = tmp_ts.shift(i)
        shift_ts_list.append(shift_ts)
        tmp_ts = tmp_ts - shift_ts
    tmp_ts.dropna(inplace=True)
    return tmp_ts

def predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_ts_list[-i-1])
            except:
                raise ValueError('What you input is not pd.Series type!')
        tmp_data.dropna(inplace=True)
    return tmp_data


if __name__ == '__main__':
    ## 解决样图中文输出乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # 读入数据
    inFileName = 'AirPassengers.csv'
    inFilePath = inout.getDataPathTimeseries(inFileName)
    data = pd.read_csv(inFilePath, header=0, index_col='Month')

    window = 3
    dif_size = 1

    ## 1 将数据按时间索引
    data.index = pd.to_datetime(data.index)
    ts = data['#Passengers'].astype(np.float)

    ## 2 时间序列平稳性处理
    # 2.1 对数处理
    ts_log = np.log(ts)
    # dif_1 = ts_log.diff(1)
    # plt.figure('white')
    # ts_log.plot(color='red',label='origin')
    # dif_1.plot(color='blue',label='dif 1')
    # plt.show()
    # exit(0)
    # train/test处理
    ts_train = ts_log[:'1956-12']
    ts_test = ts_log['1957-1':]
    # 2.2 差分处理
    diffed_ts = diff_ts(ts_train, [window, dif_size])
    order = st.arma_order_select_ic(diffed_ts, ic=['aic', 'bic', 'hqic'])
    ##
    model = ARMA(diffed_ts,order.bic_min_order).fit()
    predict_result = model.predict('1957-1','1957-12',dynamic=True)
    # print predict_result
    # exit(0)
    predict_recover = predict_diff_recover(predict_result,[window,dif_size])
    print type(predict_recover)
    print predict_recover
    exit(0)
    plt.figure('white')
    ts.plot(color='red',label='origin')
    predict_recover.plot(color='blue',label='predict')
    plt.legend(loc='best')
    plt.show()






    ## try one
    # print order.bic_min_order
    # p = order.bic_min_order[0]
    # q = order.bic_min_order[1]
    # i = 1
    # model = ARIMA(endog=ts,order=(p,i,q)).fit()
    # # summary = model.summary2()
    # # print type(summary)
    # # print summary
    # predict_result = model.forecast(15)
    # predict = pd.Series(predict_result[0])
    # print predict
    # plt.figure('white')
    # predict.plot(color='blue',label=u'预测值')
    # # ts.plot(color='red',label=u'原始值')
    # plt.legend(loc='best')
    # plt.show()