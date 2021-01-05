import math
import random
import numpy as np


def get_average(records):
    return sum(records) / len(records)


def get_variance(records):
    average = get_average(records)
    return sum([(x - average) ** 2 for x in records]) / len(records)


def get_standard_deviation(records):
    variance = get_variance(records)
    return math.sqrt(variance)


def get_z_score(records):
    avg = get_average(records)
    stan = get_standard_deviation(records)
    scores = [(i - avg) / stan for i in records]
    return scores


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loaddataset(filename):
    fp = open(filename)

    # 存放数据
    dataset = []

    # 存放标签
    labelset = []
    for i in fp.readlines():
        a = i.strip().split()

        # 每个数据行的最后一个是标签
        dataset.append([float(j) for j in a[:len(a) - 1]])
        labelset.append(int(float(a[-1])))
    return dataset, labelset


# x为输入层神经元个数，y为隐层神经元个数，z输出层神经元个数
def parameter_initialization(x, y, z):
    # 隐层阈值
    y_ = -1 / math.sqrt(y)
    sqrt_y_ = 1 / math.sqrt(y)

    value1 = []
    for i in range(y):
        value1.append(random.uniform(y_, sqrt_y_))

    # 输出层阈值
    z_ = -1 / math.sqrt(z)
    sqrt_z_ = 1 / math.sqrt(z)
    value2 = []
    for i in range(z):
        value2.append(random.uniform(z_, sqrt_z_))
    # value2 = np.random.randint(z_, sqrt_z_, (1, z)).astype(np.float64)

    # 输入层与隐层的连接权重

    weight1 = np.random.randint(-5, 5, (x, y)).astype(np.float64)
    for i in range(len(weight1)):
        for j in range(len([i])):
            weight1[j] = random.uniform(y_, sqrt_y_)
    # 隐层与输出层的连接权重
    weight2 = np.random.randint(-5, 5, (y, z)).astype(np.float64)
    for i in range(len(weight2)):
        for j in range(len([i])):
            weight2[i][j] = random.uniform(z_, sqrt_z_)
    return weight1, weight2, value1, value2


'''
weight1:输入层与隐层的连接权重
weight2:隐层与输出层的连接权重
value1:隐层阈值
value2:输出层阈值
'''


def trainning(dataset, labelset, weight1, weight2, value1, value2):
    # x为步长
    x = 0.1
    for i in range(len(dataset)):
        # 输入数据
        inputset = np.mat(dataset[i]).astype(np.float64)
        # 数据标签
        outputset = np.mat(labelset[i]).astype(np.float64)
        # 隐层输入 w * x
        input1 = np.dot(inputset, weight1).astype(np.float64)
        # 隐层输出 w * x  - b
        output2 = sigmoid(input1 + value1).astype(np.float64)
        # 输出层输入
        input2 = np.dot(output2, weight2).astype(np.float64)
        # 输出层输出
        output3 = sigmoid(input2 + value2).astype(np.float64)

        # 更新公式由矩阵运算表示 sigmod 求导 f(x) * 1 - f(x)
        g = np.multiply(np.multiply(output3, 1 - output3), outputset - output3)  # d((f(x) * 1 - f(x))) / (yi - Yi) d(Z)

        # d((f(x) * 1 - f(x))) / (yi - Yi) d(Z)
        # 下一次的b * 下一层w * f(x) * (1 - f(x))
        # 更新公式由矩阵运算表示 sigmod 求导 f(x) * 1 - f(x)
        b = np.dot(g, np.transpose(weight2))  # 下一次的b * 下一层w * f(x) * (1 - f(x))
        # 更新公式由矩阵运算表示 sigmod 求导 f(x) * 1 - f(x)
        c = np.multiply(output2, 1 - output2)
        e = np.multiply(b, c)

        value1_change = -x * e
        value2_change = -x * g
        weight1_change = x * np.dot(np.transpose(inputset), e)
        weight2_change = x * np.dot(np.transpose(output2), g)

        # 更新参数
        value1 += value1_change
        value2 += value2_change
        weight1 += weight1_change
        weight2 += weight2_change
    return weight1, weight2, value1, value2


def testing(dataset, labelset, weight1, weight2, value1, value2):
    # 记录预测正确的个数
    rightcount = 0
    for i in range(len(dataset)):
        # 计算每一个样例通过该神经网路后的预测值
        inputset = np.mat(dataset[i]).astype(np.float64)
        outputset = np.mat(labelset[i]).astype(np.float64)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)
        output3 = sigmoid(np.dot(output2, weight2) - value2)

        # 确定其预测标签
        if output3 > 0.5:
            flag = 1
        else:
            flag = 0
        if labelset[i] == flag:
            rightcount += 1
        # 输出预测结果
        # print("预测为%d 实际为%d" % (flag, labelset[i]))
    # 返回正确率
    return rightcount / len(dataset)


if __name__ == '__main__':
    bz = 0
    while bz <= 80:
        dataset, labelset = loaddataset('horseColicTraining.txt')
        i = len(dataset[0])  # 21个维度
        weight1, weight2, value1, value2 = parameter_initialization(i, i+5, 1)
        dataset = np.array(dataset)
        mean = dataset.mean()  # 计算平均数
        deviation = dataset.std()  # 计算标准差
        # 标准化数据的公式: (数据值 - 平均数) / 标准差
        dataset = (dataset - mean) / deviation
        for i in range(10):
            weight1, weight2, value1, value2 = trainning(dataset, labelset, weight1, weight2, value1, value2)
        rate = testing(dataset, labelset, weight1, weight2, value1, value2)
        bz = rate * 100
        if bz > 70:
            print("正确率为%f" % (rate))

    quit()
