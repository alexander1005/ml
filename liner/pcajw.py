import numpy as np

# 主城成分分析 最大方差投影
i = zip([10, 11, 8, 3, 2, 1],
        [6, 4, 5, 3, 2.8, 1])
list = []
for a in i:
    list.append(a)
dataset = np.mat(list)
mean = np.mean(dataset,axis=0)
data = dataset - mean
jh= np.cov(data.T)

c = data.T * data *(1/(6-1))

# 求特征向量
a,b=np.linalg.eig(c)
print(b)
# 中心化的数组 * 特征向量
c = data * b[0].T
 # 新的特征c
quit()