#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf

# # tensoflow 中tf.multiply(a,b)与tf.matmul(c,d)比较
# a = tf.constant([1.0,2.0]) # 定义一个变量a，并初始化
# b = tf.constant([3.0,4.0]) # 定义一个变量b ，并初始化
#
# c = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
# d = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) # 定义一个变量d ，并初始化
#
#
# product1 = tf.multiply(a,b) # 对应位置相乘
# product2 = tf.matmul(c,d) # 矩阵的乘法，得到的结果是一个矩阵，行乘列作为对应位置的元素
#
# with tf.Session() as sess:
#     print(sess.run(product1))
#     print(sess.run(product2))

