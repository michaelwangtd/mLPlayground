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
    ts = np.log(data)
    # plt.figure('white')
    # ts.plot()
    # plt.show()
    # exit(0)
    ts_train = ts[:80]
    ts_test = ts[80:]
    # print ts_train
    # print len(ts_train)
    # print len(ts_test)
    # print ts_test
    plt.figure('white')
    ts_train.plot(color='blue',label='ts_train')
    plt.legend(loc='best')
    plt.show()
    # 3 平稳性处理
    rollingsize = 7
    # rol_mean_real = ts_train.rolling(window=rollingsize).mean()
    # rol_mean_real.dropna(inplace=True)
    # plt.figure('white')
    # rol_mean_real.plot(color='blue',label='rol_mean_real')
    # plt.legend(loc='best')
    # plt.show()
    # exit(0)
    # diff_ts_1 = rol_mean_real.diff(1)
    # diff_ts_1.dropna(inplace=True)
    # 一阶差分acf图和adf检测值
    # adftest = adfuller(diff_ts_1)
    # print 'adftest diff_1 p-value:',adftest[1]
    # plt.figure('white')
    # diff_ts_1.plot(color='blue',label='diff_1')
    # plt.legend(loc='best')
    # plot_acf(diff_ts_1,lags=31)
    # plt.show()
    # exit(0)
    # diff_ts_2 = diff_ts_1.diff(1)
    # diff_ts_2.dropna(inplace=True)
    # 二阶差分acf图和adf检测值
    # adftest = adfuller(diff_ts_2)
    # print 'adftest_2 p-value:',adftest[1]
    # plt.figure('white')
    # diff_ts_2.plot(color='blue',label='diff_2')
    # plt.legend(loc='best')
    # plot_acf(diff_ts_2,lags=31)
    # plt.show()
    # exit(0)
    # diff_ts_3 = diff_ts_2.diff(1)
    # diff_ts_3.dropna(inplace=True)
    # adftest = adfuller(diff_ts_3)
    # print 'adftest_3 p-value:',adftest[1]
    # plt.figure('white')
    # diff_ts_3.plot(color='blue',label='diff_3')
    # plt.legend(loc='best')
    # plot_acf(diff_ts_3,lags=31)
    # plt.show()
    # exit(0)
    # 4 获取p，d值
    stable_ts = ts_train
    # print type(stable_ts)
    # print stable_ts
    # exit(0)
    # order = st.arma_order_select_ic(stable_ts, ic=['aic', 'bic', 'hqic'])
    # print order.bic_min_order
    # exit(0)
    # 5 模型
    ts_predict = []
    ts_history = [ item for item in stable_ts]
    for i in range(len(ts_test)):
        # model = ARMA(stable_ts,order.bic_min_order).fit()
        # model = ARMA(ts_history,(0,1)).fit(disp=0)
        model = ARIMA(ts_history,order=(2,2,0)).fit(disp=0)
        # model = ARMA(stable_ts,(0,1)).fit()
        # predict_ts = model.predict('2097','2106',dynamic=True)
        # print type(predict_ts)
        # print predict_ts
        output = model.forecast()
        y_hat = output[0]
        print '---y_hat: ',type(y_hat),y_hat
        ts_predict.append(y_hat[0])
        # ts_history.append(y_hat[0])
        ts_history.append(ts_test[i])
    print type(ts_predict),ts_predict

    ts_predict = np.exp(ts_predict)
    ts_predict = pd.Series(ts_predict)
    ts_test = np.exp(ts_test)

    # print len(ts_predict),type(ts_predict),ts_predict
    # print len(ts_test),type(ts_test),ts_test
    # exit(0)
    ticks = [i for i in range(len(ts_predict))]
    plt.figure('white')
    plt.plot(ticks,ts_predict,'-b',label='predic')
    plt.plot(ticks,ts_test,'-r',label='origin')
    plt.show()


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