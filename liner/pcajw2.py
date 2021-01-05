import numpy as np

# 主城成分分析 最大方差投影
i = zip([10, 11, 8, 3, 2, 1],
        [6, 4, 5, 3, 2.8, 1])

# 语文 数学 物理 化学 英语 历史
list = [
    [84, 65, 61, 72, 79, 81],
    [64, 77, 77, 76, 55, 70],
    [65, 67, 63, 49, 57, 67],
    [74, 80, 69, 75, 63, 74],
    [84, 74, 70, 80, 74, 82]
]
dataset = np.mat(list)

datamean = np.mean(dataset, axis=0)

cdata = (list - datamean).T

cov = np.cov(cdata)

a,b=np.linalg.eig(cov)
a[0] #成分
a[1] #成分
b0 = np.mat(b[0])
b1 = np.mat(b[1])
print(b0.shape)

data0 = cdata.T * b0.T
data1 = cdata.T * b1.T
print(dataset)


# list = []
# for a in i:
#     list.append(a)
# dataset = np.mat(list)
# mean = np.mean(dataset,axis=0)
# data = dataset - mean
# jh= np.cov(data.T)
#
# c = data.T * data *(1/(6-1))
#
# # 求特征向量
# a,b=np.linalg.eig(c)
# print(b)
# # 中心化的数组 * 特征向量
# c = data * b[0].T
#  # 新的特征c
quit()
