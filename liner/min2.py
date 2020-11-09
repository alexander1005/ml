import pandas
from numpy import dot  # 乘法
from numpy import mat  # 二维矩阵
from numpy.linalg import inv  # 矩阵求逆
import matplotlib.pyplot as plt
import numpy as np #
def getA(X, Y):
    return dot(dot(inv(dot(X.T, X)), X.T), Y)


X = mat([1, 2, 3]).reshape(3, 1)
Y = mat([5, 10, 15]).reshape(3, 1)
print(X)
a = getA(X, Y)
print(a)
csv = pandas.read_csv("data.csv")

Z = csv.iloc[:, 1: 5]
C = csv.iloc[:, 0]
# print(C)
D = getA(Z, C)
print(D)


# print(a[0])
y = X * 5
# y = 210(x**6)((1-x)**4) # 这里是函数的表达式
#
plt.figure() # 定义一个图像窗口
plt.plot(X, y) # 绘制曲线 y

plt.show()