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
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm


if __name__ == '__main__':
    ## 解决样图中文输出乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # 1 读入数据
    # inFileName = 'O1005-O1005_model_trial.csv'
    inFileName = 'O1002-PAX_model_trial.csv'
    inFilePath = inout.getDataPathTimeseries(inFileName)
    data = pd.read_csv(inFilePath,header=None)
    # 2 数据转换成时间序列
    data = data[4]
    start = 1900
    end = start + 90
    data.index = pd.Index(sm.tsa.datetools.dates_from_range(str(start),str(end)))
    # print type(data)
    # plt.figure('white')
    # data.plot()
    # plt.show()
    # exit(0)
    # 2 获取时间序列
    # ts = np.log(data)
    # plt.figure('white')
    # ts.plot()
    # plt.show()
    # exit(0)
    ts = data
    ts_train = ts[:80]
    ts_train = np.log(ts_train)
    ts_test = ts[80:]
    plt.figure('white')
    ts_train.plot(color='blue', label='ts_train')
    plt.legend(loc='best')
    plt.show()
    rollingsize = 7

    stable_ts = pd.Series(ts_train)
    # model = ARIMA(stable_ts, order=(2, 2, 0)).fit(disp=0)
    model = ARMA(stable_ts, (2,0)).fit(disp=0)
    # print stable_ts
    pre_start = start + 80
    pre_end = start + 90
    # print pre_start
    # print pre_end
    # exit(0)
    # y_hat = model.predict(str(pre_start),str(pre_start),dynamic=True)
    y_hat = model.forecast(str(pre_start),str(pre_start))
    print type(y_hat)