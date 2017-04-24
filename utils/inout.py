#!/usr/bin/env python
# -*- coding:utf-8 -*-

import index
import os


def getDataPathRegression(fileName):
    '''

    '''
    return os.path.join(index.rootPath,index.DATA,index.REGRESSION,fileName)


def getDataPathDecisionTree(fileName):
    '''

    '''
    return os.path.join(index.rootPath,index.DATA,index.DECISIONTREE,fileName)

