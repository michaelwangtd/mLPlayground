#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


if __name__ == '__main__':
    # 获取测试数据
    x = np.linspace(0,10,50)
    y = np.sin(x)

    x_train = x[:35].reshape(-1,1)
    y_train = y[:35]
    x_test = x[35:].reshape(-1,1)
    y_test = y[35:]

    svr_rbf = svm.SVR(kernel='rbf',gamma=0.2,C=100)
    svr_rbf.fit(x_train,y_train)
    y_hat_rbf = svr_rbf.predict(x_test)

    # svr_line = svm.SVR(kernel='linear',C=100 )
    # svr_line.fit(x_train, y_train)
    # y_hat_line = svr_line.predict(x_test)

    # svr_poly = svm.SVR(kernel='poly', degree=2, C=100 )
    # svr_poly.fit(x_train, y_train)
    # y_hat_poly = svr_poly.predict(x_test)

    plt.figure(facecolor='w')
    plt.plot(x,y,'go')
    plt.plot(x_test,y_hat_rbf,'ro')
    # plt.plot(x_test,y_hat_line,'bo')
    # plt.plot(x_test,y_hat_poly,'yo')
    plt.grid()
    plt.show()
