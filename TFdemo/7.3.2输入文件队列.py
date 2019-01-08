#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf

# 串讲TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 模拟海量数据情况下降数据写入不同的文件。num_shards定义了总共写入多少个文件
# instance_per_shard 定义了每个文件中有多少个数据
num_shards = 2
instance_per_shard = 2
for i in range(num_shards):
    # 将数据分为多个文件时，可以将不同文件以类似0000n-of-0000m的后缀区分。
    # 其中m表示了数据总共被存在了多少个文件中，n表示当前文件的编号。式样的方式既方便了通过正则表达式获取文件列表，又在文件名中加入了更多信息
    filename = ("./data/data.tfrecords-%.5d-of-%.5d" %(i,num_shards))
    writer = tf.python_io.TFRecordWriter(filename)
    # 将数据封装成Example结构并写入TFRecord文件
    for j in range(instance_per_shard):
        example = tf.train.Example(features = tf.train.Features(feature = {'i':_int64_feature(i),
                                                                          'j':_int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()

