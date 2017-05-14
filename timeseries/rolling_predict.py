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
from statsmodels.tsa import arima_model
from dateutil.relativedelta import relativedelta

"""
    结合ARIMA模型，实现滚动预测
"""

def _add_new_data(ts, dat, type='day'):
    if type == 'day':
            new_index = ts.index[-1] + relativedelta(days=1)
    elif type == 'month':
        new_index = ts.index[-1] + relativedelta(months=1)
    ts[new_index] = dat

def add_today_data(model, ts,  data, d, type='day'):
    _add_new_data(ts, data, type)  # 为原始序列添加数据
    # 为滞后序列添加新值
    d_ts = diff_ts(ts, d)
    model.add_today_data(d_ts[-1], type)

def forecast_next_day_data(model, type='day'):
    if model == None:
        raise ValueError('No model fit before')
    fc = model.forecast_next_day_value(type)
    return predict_diff_recover(fc, [12, 1])

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

    window = 12
    dif_size = 1

    ## 1 将数据按时间索引
    data.index = pd.to_datetime(data.index)
    ts = data['#Passengers']

    ## 2 时间序列平稳性处理
    # 2.1 对数处理
    ts_log = np.log(ts)
    # train/test处理
    ts_train = ts_log[:'1956-12']
    ts_test = ts_log['1957-1':]
    # 2.2 差分处理
    diffed_ts = diff_ts(ts_train,[window,dif_size])
    forecast_list = []
    # 3
    for i,data in enumerate(ts_test):
        # if i % 7 == 0:
        # 训练模型部分
        order = st.arma_order_select_ic(diffed_ts, ic=['aic', 'bic', 'hqic'])
        print order.bic_min_order
        model = ARMA(diffed_ts, order=order.bic_min_order)
        result_arma = model.fit(disp=-1, method='css')
        # predict_ts = result_arma.predict()
        forecast_data = forecast_next_day_data(model,type='Month')
        print forecast_data
        exit(0)

    """
        测试diff_ts(),predict_diff_recover()函数
    """
    data.index = pd.to_datetime(data.index)
    ts = data['#Passengers']
    ts_log = np.log(ts)
    diffed_ts = diff_ts(ts_log,[window,dif_size])
    print type(diffed_ts)

    # ## 作者封装了arima_model类
    # model = arima_model(diffed_ts)
    # model.certain_model(1, 1)
    # predict_ts = model.properModel.predict()
    # diff_recover_ts = predict_diff_recover(predict_ts, d=[12, 1])
    # log_recover = np.exp(diff_recover_ts)

    # from statsmodels.tsa.arima_model import ARMA
    # model = ARMA(ts_diff_2, order=(1, 1))
    # result_arma = model.fit(disp=-1, method='css')
    ## 传统的模型训练方法
    # # print diffed_ts
    # order = st.arma_order_select_ic(diffed_ts, ic=['aic', 'bic', 'hqic'])
    # print order.bic_min_order
    # model = ARMA(diffed_ts, order=order.bic_min_order)
    # result_arma = model.fit(disp=-1, method='css')
    # predict_ts = result_arma.predict()
    # # print predict_ts
    # diff_recover_ts = predict_diff_recover(predict_ts,d=[12,1])
    # log_recover = np.exp(diff_recover_ts)
    # plt.figure('white')  # 对原始时间序列进行预测
    # log_recover.plot(color='blue', label=u'预测值')
    # ts.plot(color='red', label=u'原始值')
    # plt.legend(loc='best')
    # plt.title('RMSE: %.4f' % np.sqrt(sum((log_recover - ts) ** 2) / ts.size))
    # plt.show()
    # print '----------------------'
    # print ts
    # print '---------------------'
    # print log_recover














