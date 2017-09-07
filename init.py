# encoding: utf-8
"""
@author: yousheng
@contact: 1197993367@qq.com
@site: http://youyuge.cn

@version: 1.0
@license: Apache Licence
@file: init.py
@time: 17/9/7 上午10:44

"""
import svmMLiA as svm
from numpy import *

def runSimpleSMO():
    dataArray,labelArray = svm.loadDataSet('testSet.txt')
    b,alphas = svm.smoSimple(dataArray,labelArray,0.6,0.001,40)
    print b
    print alphas[alphas>0]
    print shape(alphas[alphas>0])[1] #得到支持向量的个数
    for i in range(100):    #获得支持向量的坐标数据，和类别
        if alphas[i]>0.0:
            print dataArray[i],labelArray[i]


if __name__ == '__main__':
    runSimpleSMO()