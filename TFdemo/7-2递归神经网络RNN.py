#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 载入数据集
mnist = input_data.read_data_sets("./mnist", one_hot = True)

# 输入图片是28*28
n_input = 28  # 输入的一行，一行有28个数据
max_time = 28  # 一共28行
lstm_size = 100  # 隐层单元
n_classes = 10  # 10个分类
batch_size = 50  # 每批次50个样本
n_batch = mnist.train.num_examples // batch_size  # 计算一共有多少个批次

#
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

# D:/Python/Python36/Lib/site-packages/tensorflow/python/ops/rnn_cell_impl.py:645

def RNN(X, weights, biases):
    inputs = tf.reshape(X, [-1, max_time, n_input])
    # lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size) # error
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size) # is deprecated
    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size)
    out_puts, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results


prediction = RNN(x, weights, biases)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)) # is deprecated
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(11):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy = " + str(acc))
