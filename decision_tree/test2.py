#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from utils import inout
import pandas as pd
import matplotlib.pyplot as plt

def ground_truth(x):
    """Ground truth -- function to approximate"""
    return x * np.sin(x) + np.sin(2 * x)

if __name__ == '__main__':
    plt.figure(facecolor='w')

    x = np.linspace(0, 20, 100)
    y = ground_truth(x)
    x_train = np.array(x[:80]).reshape(-1,1)
    x_test = np.array(x[80:]).reshape(-1,1)
    y_train = y[:80]
    y_test = y[80:]
    gbr = GradientBoostingRegressor(n_estimators=1000, max_depth=1, learning_rate=0.1)
    gbr.fit(x_train, y_train)
    y_hat = gbr.predict(x_test)
    plt.plot(x, y, 'go')
    plt.plot(x_test, y_hat, 'ro')
    plt.grid()
    plt.show()