#!/usr/bin/env python
# -*- coding:utf-8 -*-


import tensorflow as tf

# # Fetch
# # 在会话中可以同时执行多个op（操作），得到它运行的结果
#
# # 例子定义3个常量
# input1 = tf.constant(3.0)
# input2 = tf.constant(5.0)
# input3 = tf.constant(4.0)
#
# add = tf.add(input2,input3)
# mul = tf.multiply(input1,add)
#
# with tf.Session() as sess:
#     # 同时运行多个op，一个是乘法，一个是加法，这个就是Fetch的概念
#     result = sess.run([mul,add])
#     print(result)
#
# '''
# 输出结果
# [27.0, 9.0]
# '''
#
# # Feed 这个概念很重要，以后都要用到
# # 创建占位符，可以在会话中调用它进行使用,只定义，不赋值
# input4 = tf.placeholder(tf.float32)
# input5 = tf.placeholder(tf.float32)
#
# output = tf.multiply(input4,input5)
#
# # 使用Feed，在运行时再传入，根据占位符指定的类型
# with tf.Session() as sess:
#     # Feed 是以字典的形式传入
#     print(sess.run(output,feed_dict={input4:[7.],input5:[3.]}))
# # 输出 [21.]


# 简单实例
import tensorflow as tf
import numpy as np

# 使用numpy生成100个随机点
x_data = np.random.rand(100)
# y_data 是真实值函数，斜率是0.1 截距是0.2
y_data = x_data*0.1 + 0.2

# 构造一个线性模型 用这个模型来不断优化，来预测真实y_data
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

# 二次代价函数 定义损失函数，这是均方差损失函数，
loss = tf.reduce_mean(tf.square(y_data-y))
# 定义一个梯度下降方法来进行训练 的优化器 0.2是学习率，梯度下降的步长
# 梯度下降前面有讲到
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数，训练的目的就是最小化损失函数，loss越小，预测结果越接近真实值，即b,k接近于0.2,0.1
train = optimizer.minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()

# 定于会话，训练模型
with tf.Session() as sess:
    # 需要先初始化
    sess.run(init)
    for step in range(301):
        sess.run(train)
        if step % 50 == 0:
            print(step,sess.run([k,b]))
'''
输出结果，可以看出[k,b]越来越接近真实的[0.1,0.2]
0 [0.052493513, 0.09994813]
50 [0.10139732, 0.19926457]
100 [0.100483954, 0.19974528]
150 [0.10016762, 0.19991179]
200 [0.10005805, 0.19996946]
250 [0.10002011, 0.19998941]
300 [0.100006975, 0.19999632]
'''