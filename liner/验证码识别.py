import os

import numpy as np
import tensorflow as tf
from PIL import Image

# 数据集路径
DATASET_DIR = 'C:/Users/lucki/Desktop/tensor/captcha/'

# TF 文件存放目录
TEST_DIR = 'C:/Users/lucki/Desktop/tensor/test/'


# 获取所有验证码图片
def _get_filenames_and_classes(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        path = dataset_dir + filename
        photo_filenames.append(path)
    return photo_filenames


# 图片转向量
def image_vector(data):
    list = []
    for i in data:
        list.append(np.array(i).flatten())
    return np.array(list)


# one_hot
def zero_one(data):
    c = np.zeros([10])
    c[data] = 1.0
    return np.array(c)


# 读图片
def read_image(filenames):
    image_array = []
    labels0_array = []
    labels1_array = []
    labels2_array = []
    labels3_array = []
    for i, filename in enumerate(filenames):
        image_data = Image.open(filename)
        image_data = image_data.resize((224, 224))
        image_data = np.array(image_data.convert('L')).flatten()
        labels = filename.split('/')[-1][0:4]
        labels0_array.append(zero_one(int(labels[0])))
        labels1_array.append(zero_one(int(labels[1])))
        labels2_array.append(zero_one(int(labels[2])))
        labels3_array.append(zero_one(int(labels[3])))
        image_array.append(image_data)
    return np.array(image_array), labels0_array, labels1_array, labels2_array, labels3_array


def list_split(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]


# 生成权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(x, W):
    # x为【batch,图片高，图标宽，通道数(黑白1，彩色3)】、tensor
    # w 为卷积核或滤波器【卷积核的高，卷积核的宽，输入通道数，输出通道数】
    # strides[0]= strides[3] = 1.  strides[1] 为x步长 strides[2] 为y步长
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    # ksize [0]= ksize[1] = 1  .  [1,x,y,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 训练数据集
photo_filenames = _get_filenames_and_classes(DATASET_DIR)
image, label0, label1, label2, label3 = read_image(photo_filenames[:1500])
# 测试数据集
t_photo_filenames = _get_filenames_and_classes(TEST_DIR)

t_image, t_label0, t_label1, t_label2, t_label3 = read_image(t_photo_filenames)

xdata = list_split(image, 200)
y0data = list_split(label0, 200)
y1data = list_split(label1, 200)
y2data = list_split(label2, 200)
y3data = list_split(label3, 200)

# 定义placeholer xy

x = tf.placeholder(tf.float32, [None, 50176])
y0 = tf.placeholder(tf.float32, [None, 10])
y1 = tf.placeholder(tf.float32, [None, 10])
y2 = tf.placeholder(tf.float32, [None, 10])
y3 = tf.placeholder(tf.float32, [None, 10])
# 定义变量w 和b
x_image = tf.reshape(x, [-1, 224, 224, 1])

with tf.name_scope('conv1'):
    w_conv1 = weight_variable([5, 5, 1, 256])
    b_conv1 = weight_variable([256])
    l1 = max_pool_2x2(tf.nn.sigmoid(conv2d(x_image, w_conv1) + b_conv1))  # 112 /2

with tf.name_scope('conv2'):
    w_conv2 = weight_variable([5, 5, 256, 512])
    b_conv2 = weight_variable([512])
    l2 = max_pool_2x2(tf.nn.sigmoid(conv2d(l1, w_conv2) + b_conv2))

with tf.name_scope('full1'):
    h_pool2_flag = tf.reshape(l2, [-1, 14 * 14 * 512])

    w_fl1 = weight_variable([14 * 14 * 512, 2048])
    b_fl1 = weight_variable([2048])
    h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flag, w_fl1) + b_fl1)
    # 清除无意义的神经元 减少计算
    keep_prob = tf.placeholder(tf.float32)
    fc1 = tf.nn.dropout(h_fc1, keep_prob)
with tf.name_scope('full2'):
    w_fc2 = weight_variable([2048, 512])
    b_fc2 = bias_variable([512])
    fc2 = tf.nn.relu(tf.matmul(fc1, w_fc2) + b_fc2)

with tf.name_scope('full3'):
    w_fc3 = weight_variable([512, 10])
    b_fc3 = bias_variable([10])
    predict = tf.nn.softmax(tf.matmul(fc2, w_fc3) + b_fc3)
    predict1 = tf.nn.softmax(tf.matmul(fc2, w_fc3) + b_fc3)
    predict2 = tf.nn.softmax(tf.matmul(fc2, w_fc3) + b_fc3)
    predict3 = tf.nn.softmax(tf.matmul(fc2, w_fc3) + b_fc3)

# 定义交叉熵loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y0, logits=predict))
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y1, logits=predict1))
loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y2, logits=predict2))
loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y3, logits=predict3))

# 使用AdamOptimer 进行优化
train = tf.train.AdamOptimizer(0.1).minimize(loss)
# 使用AdamOptimer 进行优化
train1 = tf.train.AdamOptimizer(0.1).minimize(loss1)
# 使用AdamOptimer 进行优化
train2 = tf.train.AdamOptimizer(0.1).minimize(loss2)
# 使用AdamOptimer 进行优化
train3 = tf.train.AdamOptimizer(0.1).minimize(loss3)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        for xd, y01, y11, y22, y33 in zip(xdata, y0data, y1data, y2data, y3data):
            t1, t2, t3, t4 = sess.run([train, train1, train2, train3],
                                      feed_dict={x: xd, y0: y01, y1: y11, y2: y22, y3: y33,keep_prob: 0.7})

        print("训练一次-- t1：" + str(t1) + " t2：" + str(t2) + " t3：" + str(t3) + " t4：" + str(t4))

        p1, p2, p3, p4 = sess.run([predict, predict1, predict2, predict3],
                                  feed_dict={x: t_image, y0: t_label0, y1: t_label1, y2: t_label2, y3: t_label3,
                                             keep_prob: 0.7})
        for i in range(len(p1)):
            def get_one(one):
                for i in range(len(one)):
                    if one[i] == 1:
                        return str(i)
                return '0'


            predict_str = get_one(p1[i]) + get_one(p2[i]) + get_one(p3[i]) + get_one(p4[i])
            act_str = get_one(t_label0[i]) + get_one(t_label1[i]) + get_one(t_label2[i]) + get_one(t_label3[i])
            print('实际为:' + act_str + '  预测为:' + predict_str)
