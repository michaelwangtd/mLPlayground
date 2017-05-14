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

"""
    这里结合http://www.cnblogs.com/foley/p/5582358.html这篇文章对时间序列类型的问题进行了尝试
    文中介绍的方法基本和时间序列相关

    如何判断序列是平稳序列？
        1）平稳序列的自相关系数会快速衰减
        2）adf单位根检验p-value太大就不能推翻序列不平稳的原假设,若该值大于0.05则显著不平稳

    ARMA模型定阶？
        1）一般ARMA中p，q阶数不超过时间序列长度的1/10
        2）利用BIC统计量动态确定

    算法总体思路：1）timeseries 2）得到平稳序列 3）根据平稳序列为ARMA模型定阶

    ARMA算法根据的是时间序列性，到现在还没有得到外延的预测值：
    date 2097-12-31 00:00:00 not in date index. Try giving a date that is in the dates index or use an integer
    只能预测已有时间索引内的值
    希望下次研究会有所进展

"""



if __name__ == '__main__':
    ## 解决样图中文输出乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # 读入数据
    inFileName = 'AirPassengers.csv'
    # inFileName = 'O1005-O1005_model_trial.csv'
    inFilePath = inout.getDataPathTimeseries(inFileName)
    data = pd.read_csv(inFilePath,header=0,index_col='Month')
    # data = pd.read_csv(inFilePath,header=None,index_col=0)

    ## 1 将数据按时间索引
    data.index = pd.to_datetime(data.index)
    ts = data['#Passengers']
    # print ts
    # print ts.head()   #读取前5列
    # print ts.head().index     #读取前5列的索引

    ## 2 时间序列平稳性处理
    tidx = data.index
    # 2.1 对数变换减小振幅
    ts_log = np.log(ts)
    print '对数时间序列：',type(ts_log)
    # 指定平滑窗口大小
    size = 12
    diff_size = 12
    # 2.2 滑动平滑与加权平滑
    rol_mean = ts_log.rolling(window=size).mean()   # 滑动平均
    rol_weighted_mean = pd.ewma(ts_log,size)    # 加权移动平均
    # 2.3 差分处理
    diff_12 = ts_log.diff(diff_size)
    diff_12.dropna(inplace=True)   # inplace:将结果付给调用函数变量本身
    diff_12_1 = diff_12.diff(1)
    diff_12_1.dropna(inplace=True)
    # print '差分结果：',diff_12_1
    # 检验差分结果的adf p值
    adftest = adfuller(diff_12_1)
    print 'adftest p_value:',adftest[1]
    # 2.4 时间序列分解
    decomposition_add = seasonal_decompose(ts_log,model='additive')     # 加法模型分解
    decomposition_multi = seasonal_decompose(ts_log,model='multiplicative')
    trend = decomposition_add.trend
    seasonal = decomposition_add.seasonal
    residual = decomposition_add.resid
    # 2.5 综合上面的方法，获得平稳序列
    rol_mean_real = ts_log.rolling(window=size).mean()
    rol_mean_real.dropna(inplace=True)
    ts_diff_1 = rol_mean_real.diff(1)
    ts_diff_1.dropna(inplace=True)
    adftest = adfuller(ts_diff_1)
    # print 'adf p-value:',adftest[1]
    ts_diff_2 = ts_diff_1.diff(1)
    # 这个是得到的平稳时间序列
    ts_diff_2.dropna(inplace=True)
    adftest = adfuller(ts_diff_2)
    print 'adf p-value:',adftest[1]
    ## 3 确定ARMA模型p，q参数
    # pmax = qmax = int(len(ts_diff_2)/10)
    # 带pmax，qmax参数的话会导致最大似然优化收敛失败
    # order = st.arma_order_select_ic(ts_diff_2,max_ar=pmax,max_ma=qmax,ic=['aic','bic','hqic'])
    order = st.arma_order_select_ic(ts_diff_2,ic=['aic','bic','hqic'])
    print type(order)
    print 'p,q参数值：',order.bic_min_order
    ## 4 拟合ARMA并预测
    print '开始拟合模型'
    model = ARMA(ts_diff_2,order=order.bic_min_order)
    result_arma = model.fit(disp=-1,method='css')
    print '模型拟合完毕'
    predict_ts = result_arma.predict()
    # print type(predict_ts)
    # print predict_ts
    ## 5 还原预测值
    diff_shift_ts = ts_diff_1.shift(1)
    diff_recover_1 = predict_ts.add(diff_shift_ts)
    rol_shift_ts = rol_mean_real.shift(1)
    diff_recover = diff_recover_1.add(rol_shift_ts)
    rol_sum = ts_log.rolling(window=11).sum()
    rol_recover = diff_recover * size - rol_sum.shift(1)
    log_recover = np.exp(rol_recover)
    # 预测值还原
    log_recover.dropna(inplace=True)
    # print log_recover
    print log_recover.index



    # exit(0)
    # plt.figure('white')     # 先画一个原始数据的acf，pacf图
    # plt.subplot(2,1,1)
    # ts.plot(color='blue',label=u'原始时间序列')
    # plt.legend(loc='best')
    # ax1 = plt.subplot(2,1,2)
    # plot_acf(ts,lags=31,ax=ax1)
    # plt.show()

    # plt.figure('white')
    # plt.subplot(4,1,1)
    # plt.plot(tidx,ts,'-g',label=u'原始时间序列')
    # plt.legend(loc='best')
    # plt.subplot(4,1,2)
    # plt.plot(tidx,ts_log,'-g',label=u'对数时间序列')
    # plt.legend(loc='best')
    # plt.subplot(4,1,3)
    # plt.plot(tidx,ts_log,'-g',label=u'对数时间序列')
    # plt.plot(tidx,rol_mean,'-r',label=u'滑动平均')
    # plt.plot(tidx,rol_weighted_mean,'-b',label=u'加权滑动平均')
    # plt.legend(loc='best')
    # plt.subplot(4,1,4)
    # plt.plot(tidx,ts_log,'-g',label=u'对数时间序列')
    # diff_12_1.plot(color='red',label=u'差分')
    # plt.legend(loc='best')
    # plt.show()

    # plt.figure('white')     # 差分结果的acf,pacf图
    # ax2 = plt.subplot(2,1,1)
    # plot_acf(diff_12_1,ax=ax2,lags=31)
    # ax3 = plt.subplot(2,1,2)
    # plot_pacf(diff_12_1,ax=ax3,lags=31)
    # plt.show()

    # plt.figure('white')
    # plt.subplot(4,1,1)
    # ts_log.plot(color='blue',label=u'对数时间序列')
    # plt.legend(loc='best')
    # plt.subplot(4,1,2)
    # trend.plot(color='blue',label=u'长期趋势')
    # plt.legend(loc='best')
    # plt.subplot(4,1,3)
    # seasonal.plot(color='blue',label=u'周期趋势')
    # plt.legend(loc='best')
    # plt.subplot(4,1,4)
    # residual.plot(color='blue',label=u'随机扰动')
    # plt.legend(loc='best')
    # plt.show()

    # plot_acf(ts_diff_2,lags=31) # 综合得到的ts_diff_2的acf图
    # plt.show()

    plt.figure('white')     # 对原始时间序列进行预测
    log_recover.plot(color='blue',label=u'预测值')
    ts.plot(color='red',label=u'原始值')
    plt.legend(loc='best')
    plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-ts)**2)/ts.size))
    plt.show()




