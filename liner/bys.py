import math
import random

import numpy as np
import pandas as pd


# 分类
def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


# 2.提取属性特征. 对一个类的所有样本,计算每个属性的均值和方差
def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarize_by_class(trainSet):
    separated = separate_by_class(trainSet)
    summaries = {}
    keyList = list(separated.keys())
    for classValue in keyList:
        summaries[classValue] = summarize(separated[classValue])
    return summaries


# 计算高斯概率密度函数. 计算样本的某一属性x的概率,归属于某个类的似然
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculate_class_probabilities(summaries, inputVector,p1,p0):
    probabilities = {}
    keyList = list(summaries.keys())
    for classValue in keyList:
        probabilities[classValue] = 1
        for i in range(len(summaries[classValue])):  # 属性个数
            mean, stdev = summaries[classValue][i]  # 训练得到的第i个属性的提取特征
            x = inputVector[i]  # 测试样本的第i个属性x
            probabilities[classValue] *= calculate_probability(x, mean, stdev)
    probabilities[0.0] = probabilities[0.0]* p0
    probabilities[1.0] = probabilities[1.0] * p1
    return probabilities


# 单个数据样本的预测. 找到最大的概率值,返回关联的类
def predict(summaries, inputVector,p1,p0):
    probabilities = calculate_class_probabilities(summaries, inputVector,p1,p0)
    bestLabel, bestProb = None, -1
    keyList = list(probabilities.keys())
    for classValue in keyList:
        if bestLabel is None or probabilities[classValue] > bestProb:
            bestProb = probabilities[classValue]
            bestLabel = classValue
    return bestLabel


# 6.计算精度
def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


# 多个数据样本的预测
def get_predictions(summaries, testSet,p1,p0):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i],p1,p0)
        predictions.append(result)
    return predictions


if __name__ == '__main__':
    filename = 'pima-indians-diabetes.csv'
    dataset = pd.read_csv(filename, header=None)
    dataset = np.array(dataset)

    # 随机划分数据:67%训练和33%测试
    trainSize = int(len(dataset) * 2 / 3)  # (512,9)(256,9)
    randomIdx = [i for i in range(len(dataset))]
    random.shuffle(randomIdx)
    trainSet = []
    testSet = []
    trainSet.extend(dataset[idx, :] for idx in randomIdx[:trainSize])
    testSet.extend(dataset[idx, :] for idx in randomIdx[trainSize:])

    # 计算模型
    summaries = summarize_by_class(trainSet)

    separate_by_class(trainSet)
    mat = np.mat(trainSet)

    a = mat[:, -1]
    p1 = a.sum() / (len(a) * 1.0)
    p0 = (len(a) - a.sum()) / (len(a) * 1.0)
    # 预测
    predictions = get_predictions(summaries, testSet,p1,p0)
    accuracy = get_accuracy(testSet, predictions)
    print(('Accuracy:{0}%').format(accuracy))

    quit()