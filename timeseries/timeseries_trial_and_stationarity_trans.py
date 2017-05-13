#/usr/bin/env python
#! -*- coding:utf-8 -*-

from utils import inout
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

"""
    这里结合http://www.cnblogs.com/foley/p/5582358.html这篇文章对时间序列类型的问题进行了尝试
    文中介绍的方法基本和时间序列相关
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
    # 2.2 滑动平滑与加权平滑
    rol_mean = ts_log.rolling(window=size).mean()   # 滑动平均
    rol_weighted_mean = pd.ewma(ts_log,size)    # 加权移动平均
    # 2.3 差分处理
    diff_12 = ts_log.diff(12)
    # print type(diff_12)
    # print diff_12
    diff_12.dropna(inplace=True)   # inplace:将结果付给调用函数变量本身
    diff_12_1 = diff_12.diff(1)
    diff_12_1.dropna(inplace=True)
    # print '差分结果：',diff_12_1
    # 2.4 时间序列分解
    decomposition_add = seasonal_decompose(ts_log,model='additive')     # 加法模型分解
    decomposition_multi = seasonal_decompose(ts_log,model='multiplicative')
    trend = decomposition_add.trend
    seasonal = decomposition_add.seasonal
    residual = decomposition_add.resid
    # print type(trend)

    ## 窗口图一
    plt.figure('white')
    plt.subplot(4,1,1)
    plt.plot(tidx,ts,'-g',label=u'原始时间序列')
    plt.legend(loc='best')
    plt.subplot(4,1,2)
    plt.plot(tidx,ts_log,'-g',label=u'对数时间序列')
    plt.legend(loc='best')
    plt.subplot(4,1,3)
    plt.plot(tidx,ts_log,'-g',label=u'对数时间序列')
    plt.plot(tidx,rol_mean,'-r',label=u'滑动平均')
    plt.plot(tidx,rol_weighted_mean,'-b',label=u'加权滑动平均')
    plt.legend(loc='best')
    plt.subplot(4,1,4)
    plt.plot(tidx,ts_log,'-g',label=u'对数时间序列')
    diff_12_1.plot(color='red',label=u'差分')
    plt.legend(loc='best')
    plt.show()
    ## 窗口图二
    plt.figure('white')
    plt.subplot(4,1,1)
    ts_log.plot(color='blue',label=u'对数时间序列')
    plt.legend(loc='best')
    plt.subplot(4,1,2)
    trend.plot(color='blue',label=u'长期趋势')
    plt.legend(loc='best')
    plt.subplot(4,1,3)
    seasonal.plot(color='blue',label=u'周期趋势')
    plt.legend(loc='best')
    plt.subplot(4,1,4)
    residual.plot(color='blue',label=u'随机扰动')
    plt.legend(loc='best')
    plt.show()


