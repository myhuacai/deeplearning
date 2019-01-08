# -*- coding: utf-8 -*-
# 作者: changqing
# 创建时间: 2019/1/8 20:29
# 文件: 7.4.1.py
# IDE: PyCharm
# @GitHub : https://github.com/myhuacai/deeplearning.git

'''
数据集的基本使用
'''
import tensorflow as tf

# # 从一个数组创建数据集
# input_data = [1,2,3,5,8]
#
# dataset = tf.data.Dataset.from_tensor_slices(input_data)
# # 定义一个迭代器用于遍历数据集。因为上面定义的数据集没有用placeholder作为输入参数，所以这里可以使用最简单的one_shot_iterator
# interator = dataset.make_one_shot_iterator()
# # get_next()返回代表一个输入数据的张量，类似于队列的dequeue()
# x = interator.get_next()
# y = x * x
#
# with tf.Session() as sess:
#     for i in range(len(input_data)):
#         print(sess.run(y))

'''
1,定义数据集的构造方法
2，定义遍历器
3，使用get_next()方法从遍历器中读取张量，作为计算图其他部分的输入
'''

'''
读取文本中的数据集
'''
# import tensorflow as tf
#
# # 从文本中创建数据集
# input_files = ['./data/1.txt','./data/2.txt']
# dataset = tf.data.TextLineDataset(input_files)
#
# # 定义迭代器用于遍历数据集
# iterator = dataset.make_one_shot_iterator()
#
# x = iterator.get_next()
#
# with tf.Session() as sess:
#     for i in range(3):
#         print(sess.run(x))

'''
读取图像文件中的数据集，图像输入数据一般保存在TFRecord中
'''
import tensorflow as tf

def parser(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'feature1':tf.FixedLenFeature([],tf.int64),
                                           'feature2': tf.FixedLenFeature([], tf.int64)
                                       })
