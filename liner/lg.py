import matplotlib.pyplot as plt
import numpy as np
from numpy import dot  # 乘法


# 模型公式  1 / (1 + e^Wtx)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidX(w, x):
    return 1 / (1 + np.exp(-(w * x)))


def ini_data():
    data = np.loadtxt('lgdata.csv')
    dataMatIn = data[:, 0:-1]
    classLabels = data[:, -1]
    dataMatIn = np.insert(dataMatIn, 0, 1, axis=1)
    return dataMatIn, classLabels


def plotBestFIt(weights,x1,x2):
    dataMatIn, classLabels = ini_data()
    n = np.shape(dataMatIn)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    pxcord1 = [x1]
    pycord1 = [x2]
    for i in range(n):
        if classLabels[i] == 1:
            xcord1.append(dataMatIn[i][1])
            ycord1.append(dataMatIn[i][2])
        else:
            xcord2.append(dataMatIn[i][1])
            ycord2.append(dataMatIn[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    ax.scatter(pxcord1,pycord1,s=50, c='blue')
    x = np.arange(-3, 3, 0.1)
    y = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]  # matix
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def learn(X, Y, alpha=0.001, times=900):
    x = np.mat(X)
    y = np.mat(Y).T
    m, n = np.shape(x)  # 行 列
    weights = np.ones((n, 1))  # WT
    for i in range(times):
        h = y - sigmoid(x * weights)
        weights = weights + alpha * dot(x.T, h)
    return weights


if __name__ == '__main__':
    x, y = ini_data()
    w = learn(x, y)
    x1 = 0.824839
    x2 = 13.730343
    d = sigmoidX(np.mat([1, x1, x2]),w)
    print(d)
    if d> 0.5:
        print(1)
    else:
        print(0)
    plotBestFIt(w,x1,x2)
