import numpy as np

m = 8  # 8个样本
n = 2  # 2个特征
featuremat = np.random.randint(1, 100, [m, n])

mean0 = np.mean(featuremat[:, 0])  # 求第一个特征列的均值，用于数据中心化
mean1 = np.mean(featuremat[:, 1])  # 求第二个特征列的均值，用于数据中心化

cov = np.sum((featuremat[:, 0] - mean0) * (featuremat[:, 1] - mean1)) / (m - 1)
print('cov: ', cov)
print('cov matrix: ')
print(np.cov(featuremat.T))  # 计算特征列之间的协方差矩阵