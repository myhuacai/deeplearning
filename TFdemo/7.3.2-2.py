#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf

'''
以下示例，这两函数的用法
tf.train.string_input_producer()
tf.train.match_filenames_once()
'''
# 使用tf.train.match_filenames_once函数获取文件列表。

files = tf.train.match_filenames_once('./data/data.tfrecords-*')

# 通过tf.train.string_input_producter函数创建输入队列，
# 输入队列中的文件列表为 tf.train.match_filenames_once获取的文件列表。
# 这里将shuffle参数设置为False，来避免随机打乱读文件的顺序。但一般在解决真实问题时，会将shuffle参数设置为True
# filenames_queue = tf.train.string_input_producer(files,shuffle=False,num_epochs=1)
filenames_queue = tf.train.string_input_producer(files,shuffle=False)

# 读取并解析一个样本
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filenames_queue)
features = tf.parse_single_example(
    serialized=serialized_example,
    features={'i':tf.FixedLenFeature([],tf.int64),
              'j':tf.FixedLenFeature([],tf.int64)
              })
with tf.Session() as sess:
    # 虽然在本段程序中没有声明一些变量，但使用tf.train.match_filenames_once函数时需要初始化一些变量
    tf.local_variables_initializer().run()
    print(sess.run(files))
    # 声明Coordinator类来协同不同线程，并启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # 多次执行获取数据的操作
    for i in range(6):
        print(sess.run([features['i'],features['j']]))
    coord.request_stop()
    coord.join(threads)
