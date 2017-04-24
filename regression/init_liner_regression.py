#!/usr/bin/env python
# -*- coding:utf-8 -*-

from utils import inout
import pandas as pd

"""
    线性回归初步
"""
if __name__ == '__main__':
    fileName = 'Advertising.csv'

    filePath = inout.getDataPathRegression(fileName)
    initData = pd.read_csv(filePath)
    print initData