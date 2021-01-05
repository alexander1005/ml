import numpy as np
import random
import matplotlib.pyplot as plt


def sign(v):
    if v > 0:
        return 1
    else:
        return -1


def training(train_num=50):
    train_data1 = [[1, 3, 1], [2, 5, 1], [3, 8, 1], [2, 6, 1]]  # 正面数据
    train_data2 = [[3, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]]  # 负面数据
    train_data = train_data1 + train_data2
    print(train_data)
    weight = [0, 0]  # 二维默认权重
    bias = 0  # b 默认为0
    learning_rate = 0.1  # 步长
    for i in range(train_num):
        train = random.choice(train_data)
        print(i, train)
        x1, x2, y = train
        y_predict = y * sign((weight[0] * x1 + weight[1] * x2 + bias))  # 公式  yi(wx + b)
        if y_predict < 0:
            weight[0] = weight[0] + learning_rate * x1 * y
            weight[1] = weight[1] + learning_rate * x2 * y
            bias = bias + learning_rate * y
    print(weight[0], weight[1], bias)

    plt.plot(np.array(train_data1)[:, 0], np.array(train_data1)[:, 1], 'ro')
    plt.plot(np.array(train_data2)[:, 0], np.array(train_data2)[:, 1], 'bo')
    x_1 = []
    x_2 = []
    for i in range(-10, 10):
        x_1.append(i)
        x_2.append((-weight[0] * i - bias) / weight[1])
    plt.plot(x_1, x_2)
    plt.show()
    return weight, bias


if __name__ == "__main__":
    w, b = training()
    # 感知机
