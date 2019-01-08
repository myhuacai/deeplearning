#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf

#
files = tf.train.match_filenames_once('./data/data.tfrecords-*')
filenames_queue = tf.train.string_input_producer(files,shuffle=False)
# 读取并解析一个样本
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filenames_queue)
features = tf.parse_single_example(
    serialized=serialized_example,
    features={'i':tf.FixedLenFeature([],tf.int64),
              'j':tf.FixedLenFeature([],tf.int64)
              })

example,label = features['i'],features['j']
# 一个batch中样例的个数
batch_size = 2
# 组合样例的队列中最多可以存储的样例个数。这个队列如果太大，那么需要占用很多内存资源；
# 如果太小，那么操作可能会因为没有数据而被阻碍(block)，从而导致训练效率降低。
# 一般来说这个队列的大小会合每一个batch的大小相关，下面一行代码给出了设置队列大小的一种方式
capacity = 1000 + 3 * batch_size

# # 使用train.barch函数来组合样例。[example,label] 参数给出了需要组合的元素，一般example和label分别代表样本和这个样本对应的正确标签。
# # batch_size 参数给出了每个batch中样例的个数。capacity给出了队列的最大容量。
# # 当队列长度等于容量时，TensorFlow将暂停入队操作，而只是等待元素出队。当元素个数小于容量时，TensorFlow将自动重新启动入队操作
# example_batch,label_batch = tf.train.batch([example,label],batch_size=batch_size,capacity=capacity)

# 使用tf.train.shuffle_batch()函数来组合样例。
# tf.train.shuffle_batch()函数的参数大部分和tf.train.batch()相似，
# 但是min_after_dequeue参数是tf.train.shuffle_batch()函数特有的，min_after_dequeue限制了出队时队列中元素的最小个数。
# 当队列中元素太少时，随机打乱样例顺序的作用就不大了。所以tf.train.shuffle_batch()提供了限制出队时最少元素的个数来保证随机打乱顺序的作用。
# 当出队函数被调用但是队列中元素不够时，出队操作将等待更多的元素入队才会完成。
# 如果min_after_dequeue参数被设定，capacity也相应调整来满足性能需求
example_batch,label_batch =tf.train.shuffle_batch([example,label],
                                                  batch_size=batch_size,
                                                  capacity=capacity,
                                                  min_after_dequeue=30)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # 获取并打印组合之后的样例。在真实问题中，这个输出一般作为神经网络的输入
    for i in range(3):
        cur_example_batch,cur_label_batch = sess.run([example_batch,label_batch])
        print(cur_example_batch,cur_label_batch)
    coord.request_stop()
    coord.join(threads)
