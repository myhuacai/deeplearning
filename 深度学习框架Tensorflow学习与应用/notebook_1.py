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

# # 使用张量记录中间结果
# a = tf.constant([1.0,2.0]) # 定义一个变量a，并初始化
# b = tf.constant([3.0,4.0]) # 定义一个变量b ，并初始化
#
# result = a + b
# # 直接计算向量的和，这样可读性会比较差
# result = tf.constant([1.0,2.0],name='a') + tf.constant([3.0,4.0],name='b')

##############################################################################################
weights = tf.Variable(tf.random_normal([2,3],stddev=2))
# tf.random_normal()
# tf.truncated_normal()
# tf.random_uniform()
# tf.random_gamma()
# tf.zeros
##############################################################################################
# 前向传播

import tensorflow as tf

# 声明2个变量。这里通过seed参数设置了随机种子（每次运行时输出一致）
w1 = tf.Variable(tf.random_normal((2,3),stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal((3,1),stddev=1,seed=1))
# 暂时将输入的特征向量定义为一个常数。1X2的矩阵
x = tf.constant([[0.8,0.9]])
# 前向传播算法获得神经网络的输出
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# 定义Session，运行
# with tf.Session() as sess:
    # 初始化参数
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    # sess.run(tf.global_variables_initializer())
    # 运行，打印运行后结果
    # print(sess.run(y))
# [[4.2442317]]
##############################################################################################
#tensor.get_shape()
#tensor.get_shape().as_list()

# a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='a')
# with tf.Session() as sess:
#     print(a.eval())
#     print("shape: ", a.get_shape(), ",type: ", type(a.get_shape()))
#     print("shape: ", a.get_shape().as_list(), ",type: ", type(a.get_shape().as_list()))

'''
[[1. 2. 3.]
 [4. 5. 6.]]
shape:  (2, 3) ,type:  <class 'tensorflow.python.framework.tensor_shape.TensorShape'>
shape:  [2, 3] ,type:  <class 'list'>
'''

#tf.argmax
#tf.argmax(input, dimension, name=None) returns the index with the largest value across dimensions of a tensor.
# a = tf.constant([[1, 6, 5], [2, 3, 4]])
# with tf.Session() as sess:
#     print(a.eval())
#     print("argmax over axis 0")
#     print(tf.argmax(a, 0).eval())
#     print("argmax over axis 1")
#     print(tf.argmax(a, 1).eval())
'''
[[1 6 5]
 [2 3 4]]
argmax over axis 0
[1 0 0]
argmax over axis 1
[1 2]
'''
###

#tf.reduce_sum
#tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None) computes the sum of elements across dimensions of a tensor. Unless keep_dims is true, the rank of the tensor is reduced by 1 for each entry in reduction_indices. If keep_dims is true, the reduced dimensions are retained with length 1. If reduction_indices has no entries, all dimensions are reduced, and a tensor with a single element is returned

# a = tf.constant([[1, 1, 1], [2, 2, 2]])
# with tf.Session() as sess:
#     print(a.eval())
#     print("reduce_sum over entire matrix")
#     print(tf.reduce_sum(a).eval())
#     print("reduce_sum over axis 0")
#     print(tf.reduce_sum(a, 0).eval())
#     print("reduce_sum over axis 0 + keep dimensions")
#     print(tf.reduce_sum(a, 0, keep_dims=True).eval())
#     print("reduce_sum over axis 1")
#     print(tf.reduce_sum(a, 1).eval())
#     print("reduce_sum over axis 1 + keep dimensions")
#     print(tf.reduce_sum(a, 1, keep_dims=True).eval())
'''
[[1 1 1]
 [2 2 2]]
reduce_sum over entire matrix
9
reduce_sum over axis 0
[3 3 3]
reduce_sum over axis 0 + keep dimensions
[[3 3 3]]
reduce_sum over axis 1
[3 6]
reduce_sum over axis 1 + keep dimensions
[[3]
 [6]]
'''

# tf.Variable
# tf.Tensor.name
# tf.all_variables

# # variable will be initialized with normal distribution
# var = tf.Variable(tf.random_normal([3], stddev=0.1), name='var')
# with tf.Session() as sess:
#     print(var.name)
#     tf.initialize_all_variables().run()
#     print(var.eval())
# # var:0
# # [ 0.05310677 -0.10746826  0.01206805]
#
# var2 = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name='my_var')
# var3 = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name='my_var')
# with tf.Session() as sess:
#     print(var2.name)
#     print(var3.name)
# '''
# my_var:0
# my_var_1:0
# '''
# with tf.Session() as sess:
#     for var in tf.all_variables():
#         print(var.name)
# '''
# var:0
# my_var:0
# my_var_1:0
# '''


##############################################################################################
# '''
# tf.get_variable
#
# tf.get_variable(name, shape=None, dtype=None, initializer=None, trainable=True) is used to get or create a variable instead of a direct call to tf.Variable. It uses an initializer instead of passing the value directly, as in tf.Variable. An initializer is a function that takes the shape and provides a tensor with that shape. Here are some initializers available in TensorFlow:
# •tf.constant_initializer(value) initializes everything to the provided value,
# •tf.random_uniform_initializer(a, b) initializes uniformly from [a, b],
# •tf.random_normal_initializer(mean, stddev) initializes from the normal distribution with the given mean and standard deviation.
#
# '''
# my_initializer = tf.random_normal_initializer(mean=0, stddev=0.1)
# v = tf.get_variable('v', shape=[2, 3], initializer=my_initializer)
# with tf.Session() as sess:
#     # tf.initialize_all_variables().run()
#     tf.global_variables_initializer().run()
    # print(v.eval())
# '''
# [[ 0.14729649 -0.07507571 -0.00038549]
#  [-0.02985961 -0.01537443  0.14321376]]
# '''
##############################################################################################
# tf.variable_scope
# tf.variable_scope(scope_name) manages namespaces for names passed to tf.get_variable.
#     with tf.variable_scope('layer1'):
#         w = tf.get_variable('v', shape=[2, 3], initializer=my_initializer)
#         print(w.name)
#
#     with tf.variable_scope('layer2'):
#         w = tf.get_variable('v', shape=[2, 3], initializer=my_initializer)
#         print(w.name)
'''
layer1/v:0
layer2/v:0
'''

# '''
# reuse_variables
# Note that you should run the cell above only once. If you run the code above more than once, an error message will be printed out: "ValueError: Variable layer1/v already exists, disallowed.". This is because we used tf.get_variable above, and this function doesn't allow creating variables with the existing names. We can solve this problem by using scope.reuse_variables() to get preivously created variables instead of creating new ones.
# '''
# my_initializer = tf.random_normal_initializer(mean=0, stddev=0.1)
# v = tf.get_variable('v', shape=[2, 3], initializer=my_initializer)
#
# with tf.variable_scope('layer1'):
#     w = tf.get_variable('v', shape=[2, 3], initializer=my_initializer)
#     # print(w.name)
#
# with tf.variable_scope('layer2'):
#     w = tf.get_variable('v', shape=[2, 3], initializer=my_initializer)
#     # print(w.name)
#
# with tf.variable_scope('layer1', reuse=True):
#     w = tf.get_variable('v')   # Unlike above, we don't need to specify shape and initializer
#     print(w.name)
#
#     # or
# with tf.variable_scope('layer1') as scope:
#     scope.reuse_variables()
#     w = tf.get_variable('v')
#     print(w.name)

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

add = tf.add(x, y)
multiply = tf.multiply(x, y)

# feed_dict 这个后面会讲，这里简单理解为对之前的占位符变量赋值
with tf.Session() as sess:
    print("2 + 3 = %d" % sess.run(add, feed_dict={x: 2, y: 3}))
    print("3 x 4 = %d" % sess.run(multiply, feed_dict={x: 3, y: 4}))
