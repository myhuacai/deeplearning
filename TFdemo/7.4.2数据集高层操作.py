#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf

# 列举输入文件。训练集和测试使用不同的数据
train_files = tf.train.match_filenames_once("./data/train_file-*")
test_files = tf.train.match_filenames_once("./data/test_file-*")

# 定义parser方法从TFRecord中解析数据。这里假设image中存储的是图像的原始数据，
# label是改样例所对应的标签。height,width和channels给出了图片的维度

def parser(record):
    features = tf.parse_single_example(
        record,features={
            'image':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64),
            'height':tf.FixedLenFeature([],tf.int64),
            'width':tf.FixedLenFeature([],tf.int64),
            'channels':tf.FixedLenFeature([],tf.int64)
        }
    )
    # 从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
    decoded_image = tf.decode_raw(features['image'],tf.uint8)
    decoded_image.set_shape(features['height'],features['width'],features['channels'])
    label = features['label']
    return decoded_image,label

image_size = 299 # 定义神经网络输入层图片的大小
batch_size = 100 # 定义组合数据batch的大小
shuffle_buffer = 10000 # 定义随机打乱数据时buffer的大小

# 定义读取训练数据的数据集
dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parser)

# 对数据依次进程预处理，shuffle和batching操作







# 未完待续...

