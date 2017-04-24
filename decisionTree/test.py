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


def gen_data(n_samples=200):
    """generate training and testing data"""
    np.random.seed(13)
    x = np.random.uniform(0, 10, size=n_samples)
    x.sort()
    y = ground_truth(x) + 0.75 * np.random.normal(size=n_samples)
    train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
    x_train, y_train = x[train_mask, np.newaxis], y[train_mask]
    x_test, y_test = x[~train_mask, np.newaxis], y[~train_mask]
    return x_train, x_test, y_train, y_test





def plot_data(figsize=(8, 5)):
    fig = plt.figure(facecolor='w')
    plt.plot(x_plot, ground_truth(x_plot),alpha=0.4, label='ground truth')

    # plot training and testing data
    plt.scatter(X_train, y_train, s=10, alpha=0.4)
    plt.scatter(X_test, y_test, s=10, alpha=0.4, color='red')
    plt.xlim((0, 10))
    plt.ylabel('y')
    plt.xlabel('x')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = gen_data(200)

    # plot ground truth
    x_plot = np.linspace(0, 10, 500)

    plot_data(figsize=(8, 5))

    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=1)
    gbr.fit(X_train,y_train)
    y_hat = gbr.predict(X_test)
    plt.figure(facecolor='w')
    plt.plot(X_train, y_train, 'go')
    plt.plot(X_test, y_hat, 'ro')
    plt.grid()
    plt.show()