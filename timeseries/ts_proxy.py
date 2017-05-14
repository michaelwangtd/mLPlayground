#/usr/bin/env python
#! -*- coding:utf-8 -*-

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
import statsmodels.api as sm


if __name__ == '__main__':
    ## 解决样图中文输出乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # 1 读入数据
    inFileName = 'O1005-O1005_model_trial.csv'
    inFilePath = inout.getDataPathTimeseries(inFileName)
    data = pd.read_csv(inFilePath,header=None)
    print data.head()
    # 2 获取时间序列
    ts = data[4].astype(float)
    ts.index = pd.Index(sm.tsa.datetools.dates_from_range('2016','2106'))
    # ts.index = pd.Index(data[0])
    # ts.index = pd.to_datetime(ts.index)
    # print ts.head()
    # print type(ts)
    # print ts.index
    # exit(0)
    ts_train = ts[:80]
    ts_test = ts[81:]
    # 3 平稳性处理
    ts_log = np.log(ts_train)
    diff_ts_1 = ts_log.diff(1)
    diff_ts_1.dropna(inplace=True)
    # 一阶差分acf图和adf检测值
    adftest = adfuller(diff_ts_1)
    print 'adftest p-value:',adftest[1]
    plot_acf(diff_ts_1,lags=31)
    plt.show()
    # diff_ts_2 = diff_ts_1.diff(1)
    # diff_ts_2.dropna(inplace=True)
    # # 二阶差分acf图和adf检测值
    # adftest = adfuller(diff_ts_2)
    # print 'adftest p-value:',adftest[1]
    # plot_acf(diff_ts_2,lags=31)
    # plt.show()
    # 4 获取p，d值
    stable_ts = diff_ts_1
    # print type(stable_ts)
    # print stable_ts
    # exit(0)
    # order = st.arma_order_select_ic(stable_ts, ic=['aic', 'bic', 'hqic'])
    # print order.bic_min_order
    # 5 模型
    # model = ARMA(stable_ts,order.bic_min_order).fit()
    model = ARMA(stable_ts,(0,1)).fit()
    predict_ts = model.predict('2097','2106',dynamic=True)
    print type(predict_ts)
    print predict_ts




    # plt.figure('white')
    # ax1 = plt.subplot(4,1,1)
    # ts.plot(ax=ax1,color='blue',label='origin')
    # plt.legend(loc='best')
    # ax2 = plt.subplot(4,1,2)
    # ts_log.plot(ax=ax2,color='blue',label='log')
    # plt.legend(loc='best')
    # ax3 = plt.subplot(4,1,3)
    # diff_ts_1.plot(ax=ax3,color='blue',label='diff_1')
    # plt.legend(loc='best')
    # # ax4 = plt.subplot(4,1,4)
    # # diff_ts_2.plot(ax=ax4,color='blue',label='diff_2')
    # # plt.legend(loc='best')
    # plt.show()