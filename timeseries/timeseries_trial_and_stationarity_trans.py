#/usr/bin/env python
#! -*- coding:utf-8 -*-

from utils import inout
import pandas as pd

"""
    这里结合http://www.cnblogs.com/foley/p/5582358.html这篇文章对时间序列类型的问题进行了尝试
    文中介绍的方法基本和时间序列相关
"""

if __name__ == '__main__':
    # 读入数据
    inFileName = 'AirPassengers.csv'
    # inFileName = 'O1005-O1005_model_trial.csv'
    inFilePath = inout.getDataPathTimeseries(inFileName)
    data = pd.read_csv(inFilePath,header=0,index_col='Month')
    # 将字符串索引换成时间索引
    data.index = pd.to_datetime(data.index)
    ts = data['#Passengers']
    # print ts
    # print ts.head()
    # print ts.head().index