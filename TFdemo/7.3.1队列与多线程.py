#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf


# 创建一个先进先出队列，指定队列中最多可以保存两个元素，并指定类型为整数
q = tf.FIFOQueue(2,"int32")
# 使用enqueue_many函数来初始化队列中的元素。和变量初始化话类似，在使用队列之前需要明确的调用这个初始化过程
init = q.enqueue_many(([0,10],))
# 使用Dequeue函数将队列中的第一个元素出队列
x = q.dequeue()
# 将得到的值加1
y = x + 1
# 将加1后的值再重新加入队列
q_inc = q.enqueue([y])

with tf.Session() as sess:
    # 运行初始化队列的操作
    init.run()
    for _ in range(6):
        # 运行q_inc将执行数据出队列、出队的元素+1、重新加入队列的整个过程
        v,_ = sess.run([x,q_inc])
        # 打印元素的取值
        print(v)
# dfsdfds