#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf

# 声明一个先进先出的队列，队列中最多100个元素，类型为实数
queue = tf.FIFOQueue(100,"float")
# 定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

# print(enqueue_op)
'''
def random_normal(shape,
                  mean=0.0,
                  stddev=1.0,
                  dtype=dtypes.float32,
                  seed=None,
                  name=None):
'''
# 使用tf.train.QueeuRunner 来创建多个线程运行队列的入队操作
# tf.train.QueueRunner 的第一个参数给出了被操作的队列，[enqueue_op]*5，表示了需要启动5个线程，每个线程中运行的是enqueue_op
qr = tf.train.QueueRunner(queue,[enqueue_op]*5)

# 将定义过的QueueRunner加入TensorFlow计算图上指定的集合
# tf.train.add_queue_runner函数没有指定集合，则加入默认集合tf.GrapgKeys.QUEUE_RUNNERS
# 下面的函数就是刚刚定义的
# qr加入默认的tf.GraphKeyers.QUEUE_RUNNERS集合
tf.train.add_queue_runner(qr)
# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # print(sess.run([tf.random_normal([1])] * 5))
    # print(sess.run(tf.random_normal([1])))
    # print("-------------------")
    # 使用tf.train,Coordinator()来协同启动的线程
    coord = tf.train.Coordinator()
    # 使用tf.train.QueueRunner时，需要明确调用tf.train.start_queue_runner
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for _ in range(3):
        print(sess.run(out_tensor)[0])
    coord.request_stop()
    coord.join(threads)

# sss test