#coding=utf-8

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt

#kNN算法核心
def classify0(inX, dataSet, labels, k):#inX是用于分类的输入向量，dataSet是输入的训练!!样本集，k是选择最近邻居的数目
    dataSetSize = dataSet.shape[0]#行数
    diffMat = np.tile(inX, (dataSetSize, 1)) -  dataSet #Numpy的 tile() 函数，就是将原矩阵横向、纵向地复制,dataSetSize行，1列,求每一个样本和inX的距离
    sqDiffMat = diffMat**2#双星表示每一个元素平方
    sqDistances = sqDiffMat.sum(axis=1)#sum(axis=1)，就是将一个矩阵的每一行向量相加
    distances = sqDistances**0.5#双星表示各个元素开平方
    sortedDistIndicies = distances.argsort()#argsort函数返回的是数组值从小到大顺序排列的索引值
    classCount={} #定义一个空字典
    for i in range(k):#统计前K个样本，各个label出现的次数
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)#key主要是用来进行比较的元素,b=operator.itemgetter(2) 义函数b，获取对象的第一个域的值
    return sortedClassCount[0][0]

#将文本文件转换成numpy矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)#文件行数
    datingDataMat = np.zeros((numberOfLines,3))#得到全0矩阵，numberOfLines行，3列
    datingLabels = []#将列表得最后一列储存到向量classLabelVector中
    index = 0
    for line in arrayOlines:
        line = line.strip()#截取所有回车字符,strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）
        listFromLine = line.split('\t')#变成列表
        datingDataMat[index,:] = listFromLine[0:3]
        datingLabels.append(int(listFromLine[-1]))#存储最后一行label列
        index += 1
    return datingDataMat, datingLabels

#数据可视化 给散点图加legend图例，思路是把三种不同标签的图分开，分成三个子图，画在一张图里面
def showLegendData(datingDataMat,datingLabels):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    type1 = []
    type2 = []
    type3 = []
    for i in range(len(datingLabels)):
        if datingLabels[i] == 1:
            type1.append(np.array(datingDataMat[i]))
        elif datingLabels[i] == 2:
            type2.append(np.array(datingDataMat[i]))
        else:
            type3.append(np.array(datingDataMat[i]))
    type1 = np.array(type1)#转换成array数组
    type2 = np.array(type2)
    type3 = np.array(type3)
    # 此处直接使用type1就会出现list indices must be integers or slices, not tuple错误
    # 因为type1的类型是list，需要转换为numpy的ndarray
    g1 = plt.scatter(type1[:, 0], type1[:, 1], c='red')#X[:,1] 就是取所有行的第1列的元素，scatter参数需要是数组而不是列表
    g2 = plt.scatter(type2[:, 0], type2[:, 1], c='yellow')
    g3 = plt.scatter(type3[:, 0], type3[:, 1], c='blue')
    plt.legend(handles=[g1, g2, g3], labels=['not at all', 'a small doses', 'a large doses'])
    plt.title('每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    plt.xlabel('每年获得的飞行常客里程数')
    plt.ylabel('玩视频游戏所消耗时间占比')
    plt.show()

#归一化特征值核心公式 newvalue=(oldvalue-min)/(max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    range = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0] #shape[0]返回的是dataset这个array的行数
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(range,(m,1))
    return normDataSet

#根据数值预测分类
def datingClassTest(normMat,datingLabels):
    ratio = 0.1 #90%训练样本，10%测试样本
    normMat1 = np.array(normMat)#将list转换为array格式
    datingLabels1 = np.array(datingLabels)
    n = normMat1.shape[0]
    numTestVecs = int(n * ratio) #测试样本个数
    errorCount = 1
    for i in range(numTestVecs):
        ClassifierResult = classify0(normMat1[i, :], normMat1[numTestVecs:n, :], datingLabels1[numTestVecs:n], 3)
        print('测试分类为%d，实际分类为%d'%(ClassifierResult,datingLabels1[i]))
        if(ClassifierResult!=datingLabels1[i]):
            errorCount += 1
    print('分类器错误率为%f'%(errorCount/float(numTestVecs)))# %f表示结果为浮点型 %表示十进制整数 %s表示字符串
    print(type(errorCount))
    print(type(numTestVecs))

if __name__ == '__main__':
    filename = 'datingTestSet2.txt'
    datingDataMat, datingLabels = file2matrix(filename)
    normMat = autoNorm(datingDataMat)
    showLegendData(normMat, datingLabels)
    datingClassTest(normMat, datingLabels)