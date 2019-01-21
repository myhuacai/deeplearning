#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf # 几乎默认这种

# a = tf.constant([1.0,2.0]) # 定义一个变量a，并初始化
# b = tf.constant([3.0,4.0]) # 定义一个变量b ，并初始化
# add = a + b # 定义计算 (这时并没有真正计算，只是定义了计算图逻辑)
# print(add) # 打印出 add 从结果看，这也只是一种定义没有真正计算出a+b的结果，
# # 三个节点 a , b , add 。注意这里没有显式的指定计算图(tf.Graph),则会通过tf.get_default_graph()获取默认的计算图
# # 输出  Tensor("add:0", shape=(2,), dtype=float32)
# print(a.graph is tf.get_default_graph())
# # 输出 True
# product = tf.multiply(a,b)
#
# # 下面定义不同的计算图，并执行计算
# import tensorflow as tf
#
# g1 = tf.Graph()
# with g1.as_default():
#     # 在计算图g1中定义变量v，并初始化为0
#     v = tf.get_variable('v', shape=[1], initializer=tf.zeros_initializer())
#     # # 初始化为常数
#     # v = tf.get_variable('v',initializer=tf.constant([2.0,3.1]))
#
#
# g2 = tf.Graph()
# with g2.as_default():
#     # 在计算图g2中定义变量v，并初始化为1
#     v = tf.get_variable('v', shape=[1], initializer=tf.ones_initializer())
#
#
#
# # 在计算图g1 中读取变量 'v'的值，输出
# with tf.Session(graph=g1) as sess:  # 这种也几乎是默认，参考官网
#     # TensorFlow中，run之前需要先初始化
#     tf.global_variables_initializer().run()
#     with tf.variable_scope('',reuse=True):
#         # 根据计算图g1初始化的值输出
#         print(sess.run(tf.get_variable('v')))
#         # tf.get_variable('v') 获取变量，因前面指定了计算图g1，这里会到g1中获取
#
# # 在计算图g2 中读取变量 'v'的值，输出
# with tf.Session(graph=g2) as sess:
#     # 初始化
#     tf.global_variables_initializer().run()
#     with tf.variable_scope('',reuse=True):
#         # 根据计算图g2初始化的值输出，这行会输出[1.]
#         print(sess.run(tf.get_variable('v')))

##############################################################################################

# tensoflow 中tf.multiply(a,b)与tf.matmul(c,d)比较
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

##############################################################################################



















