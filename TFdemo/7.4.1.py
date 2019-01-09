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
# import tensorflow as tf
#
# # 解析一个TFRecord的方法。record是从文件中读取的一个样例。
# def parser(record):
#     features = tf.parse_single_example(record,
#                                        features={
#                                            'feature1':tf.FixedLenFeature([],tf.int64),
#                                            'feature2': tf.FixedLenFeature([], tf.int64)
#                                        })
#     return features['feature1'],features['feature2']
# # 从TFRecord文件创建数据集
# input_files = ["./input_file1","./input_file2"]
# dataset = tf.data.TFRecordDataset(input_files)
# # map()函数表示对数据集中的每一条数据进行调用响应的方法。使用TFRecordDataset读出的是二进制的数据，
# # 这里需要通过map()来调用parser()对二进制数据进行解析，类似的，map()函数也可以用来完成其他的数据预处理工作
# dataset = dataset.map(parser)
#
# # 定义遍历数据集的迭代器,遍历数据集。在使用make_one_shot_iterator()时，数据集的所有参数必须已经确定，
# # 因此make_one_shot_iterator()不需要特别的初始化过程
# iterator = dataset.make_one_shot_iterator()
#
# # feat1,feat2是parser()返回的一维int64型张量，可以作为输入用于进一步的计算
# feat1,feat2 = iterator.get_next()
#
# with tf.Session() as sess:
#     for i in range(10):
#         f1,f2 = sess.run([feat1,feat2])

'''
使用initializable_iterator来动态初始化数据集
'''
import tensorflow as tf

# 解析一个TFRecord的方法，同上
def parser(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'feature1':tf.FixedLenFeature([],tf.int64),
                                           'feature2': tf.FixedLenFeature([], tf.int64)
                                       })
    return features['feature1'],features['feature2']

# 从TFRecord文件创建数据集，具体文件路径是一个placeholder，稍后再提供具体路径
input_files = tf.placeholder(tf.string)
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(parser)

# 定义遍历dataset的initializable_iterator
iterator = dataset.make_initializable_iterator()

feat1,feat2 = iterator.get_next()

with tf.Session() as sess:
    # 初始化iterator，并给出input_files的值
    sess.run(iterator.initializer,feed_dict={
        input_files:["./data/file1","./data/file2"]
    })
    # 遍历所有数据一个epoch。当遍历结束时，程序会抛出OutOfRangeError
    while True:
        try:
            sess.run([feat1,feat2])
        except tf.errors.OutOfRangeError:
            break





