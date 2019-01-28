#!/usr/bin/env python
# -*- coding:utf-8 -*-


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# In[3]:

# 载入数据集
mnist = input_data.read_data_sets("TFdemo/mnist", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))

'''
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting TFdemo/mnist\train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting TFdemo/mnist\train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting TFdemo/mnist\t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting TFdemo/mnist\t10k-labels-idx1-ubyte.gz
2019-01-28 14:23:48.031093: I C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
Iter 0,Testing Accuracy 0.8303
Iter 1,Testing Accuracy 0.8707
Iter 2,Testing Accuracy 0.8819
Iter 3,Testing Accuracy 0.8874
Iter 4,Testing Accuracy 0.8941
Iter 5,Testing Accuracy 0.8968
Iter 6,Testing Accuracy 0.8986
Iter 7,Testing Accuracy 0.902
Iter 8,Testing Accuracy 0.9037
Iter 9,Testing Accuracy 0.9056
Iter 10,Testing Accuracy 0.9063
Iter 11,Testing Accuracy 0.9072
Iter 12,Testing Accuracy 0.9076
Iter 13,Testing Accuracy 0.91
Iter 14,Testing Accuracy 0.91
Iter 15,Testing Accuracy 0.9113
Iter 16,Testing Accuracy 0.9118
Iter 17,Testing Accuracy 0.9124
Iter 18,Testing Accuracy 0.913
Iter 19,Testing Accuracy 0.913
Iter 20,Testing Accuracy 0.914
'''