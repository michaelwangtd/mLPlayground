#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from utils import inout
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # x_plot = np.linspace(0, 10, 50)
    # print x_plot
    # print x_plot[:,np.newaxis]

    fileName = 'O1014-O1014.xlsx'
    filePath = inout.getDataPathDecisionTree(fileName)
    initData = pd.read_excel(filePath)
    x = np.array(initData['day'])
    y = np.array(initData['avg'])
    x_train = x[:80].reshape(-1,1)
    y_train = y[:80]
    x_test = x[80:].reshape(-1,1)
    y_test = y[80:]
    gbr = GradientBoostingRegressor(n_estimators=100,max_depth=3,learning_rate=1)
    gbr.fit(x_train,y_train)
    y_hat = gbr.predict(x_test)
    print y_hat
    plt.figure(facecolor='w')
    plt.plot(x, y, 'go')
    plt.plot(x_test, y_hat, 'ro')
    plt.grid()
    plt.show()






