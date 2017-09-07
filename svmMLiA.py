# encoding: utf-8
"""
@author: yousheng
@contact: 1197993367@qq.com
@site: http://youyuge.cn

@version: 1.0
@license: Apache Licence
@file: svmMLiA.py
@time: 17/9/7 上午10:31

"""
from numpy import *
from time import sleep

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    简化版的SMO算法，遍历选取第一个alpha，在剩下的集合中随机选另一个alpha，然后同时改变
    :param dataMatIn: 输入数据
    :param classLabels: 标记
    :param C: 常数
    :param toler: 容错率
    :param maxIter: 迭代次数
    :return: b,alpha
    '''
    dataMatrix = mat(dataMatIn);   # 转换成矩阵
    labelMat = mat(classLabels).transpose()  # 转换成矩阵，并转置，标记成为一个列向量，每一行和数据矩阵对应
    m,n = shape(dataMatrix)  # 行，列

    b = 0;  # 参数b的初始化
    alphas = mat(zeros((m,1)))  # 参数alphas是个list，初始化也是全0，大小等于样本数
    iter = 0  # 当前迭代次数，maxIter是最大迭代次数

    while (iter < maxIter):  # 当超过最大迭代次数，退出
        alphaPairsChanged = 0  # 标记位，记录alpha在该次循环中，有没有优化
        for i in range(m):  # 第i个样本
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b  # 第i样本的预测类别
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions  # 误差

            #是否可以继续优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)  # 随机选择第j个样本
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b  # 样本j的预测类别
                Ej = fXj - float(labelMat[j])  # 误差

                alphaIold = alphas[i].copy()  # 拷贝，分配新的内存
                alphaJold = alphas[j].copy()

                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L==H: print "L==H"; continue
                #eta是alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T

                if eta >= 0: print "eta>=0"; continue

                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)  # 门限函数阻止alpha_j的修改量过大

                #如果修改量很微小
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue

                # alpha_i的修改方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                # 为两个alpha设置常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0

                # 说明alpha已经发生改变
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)

        #如果没有更新，那么继续迭代；如果有更新，那么迭代次数归0，继续优化
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter

    # 只有当迭代了最大次数，且alpha都不再更新了，这个时候才返回优化之后的alpha和b
    return b,alphas
