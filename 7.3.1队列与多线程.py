#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf


# 创建一个先进先出队列，指定队列中最多可以保存两个元素，并指定类型为整数
q = tf.FIFOQueue(2,"int32")
# 使用enqueue_many函数来初始化队列中的元素。和变量初始化话类似，在使用队列之前需要明确的调用这个初始化过程
init = q.enqueue_many(([0,10],))
# 使用Dequeue函数将队列中的第一个元素出队列


