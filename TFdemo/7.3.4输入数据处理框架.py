#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

files = tf.train.match_filenames_once("./data/output.tfrecords")
filename_queue = tf.train.string_input_producer(files,shuffle=False)


reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
# 以下书上的代码可能有误
# features = tf.parse_single_example(serialized_example,
#                                    features = {
#                                        'image':tf.FixedLenFeature([],tf.string),
#                                        'label':tf.FixedLenFeature([],tf.int64),
#                                        'height':tf.FixedLenFeature([],tf.int64),
#                                        'width':tf.FixedLenFeature([],tf.int64),
#                                        'channels':tf.FixedLenFeature([],tf.int64)
#                                    })
# image,label = features['image'],features['label']
# height,width = features['height'],features['width']
# channels = features['channels']
#
# decode_image = tf.decode_raw(image,tf.uint8)
# decode_image.set_shape[height,width,channels]

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    })

decoded_images = tf.decode_raw(features['image_raw'],tf.uint8)
retyped_images = tf.cast(decoded_images, tf.float32)
labels = tf.cast(features['label'],tf.int32)
#pixels = tf.cast(features['pixels'],tf.int32)
images = tf.reshape(retyped_images, [784])

# 以下代码有误，待整理
# # 以下两个函数是7.2.2.py中的，直接拿来用的
# def distort_color(image,color_ordering = 0):
#     if color_ordering == 0:
#         image = tf.image.random_brightness(image,max_delta=32./255)
#         image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
#         image  = tf.image.random_hue(image,max_delta=0.2)
#         image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
#     elif color_ordering == 1:
#         image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
#         image = tf.image.random_brightness(image,max_delta=32./255)
#         image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
#         image  = tf.image.random_hue(image,max_delta=0.2)
#     elif color_ordering == 2:
#         image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
#         image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
#         image = tf.image.random_hue(image, max_delta=0.2)
#         image = tf.image.random_brightness(image, max_delta=32. / 255)
#         # 还可以添加其他的排列
#     else:
#         image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
#         image = tf.image.random_brightness(image,max_delta=32./255)
#         image = tf.image.random_hue(image, max_delta=0.2)
#         image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
#     return tf.clip_by_value(image,0.0,1.0)
# def preprocess_for_train(image,height,width,bbox):
#     # 如果没有提供标准框，则认为整个图像就是需要关注的部分
#     if bbox is None:
#         bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
#     # 转换图像张量的类型：
#     if image.dtype != tf.float32:
#         image = tf.image.convert_image_dtype(image,dtype=tf.float32)
#     # 随机截取图像，减少需要关注的物体大小对图像识别算法的影响
#     bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
#     distorted_image = tf.slice(image,bbox_begin,bbox_size)
#     # 将随机截取的图像调整为神经网络输入层的大小。大小调整的算法是随机选择的。
#     distorted_image = tf.image.resize_images(distorted_image,[height,width],method=np.random.randint(4))
#     # 随机左右翻转图像
#     distorted_image = tf.image.random_flip_left_right(distorted_image)
#     # 使用一种随机的顺序处理图像的色彩
#     distorted_image = distort_color(distorted_image,np.random.randint(2))
#     return  distorted_image
# # 定义神经网络输入层图片的大小
# image_size = 299
# distored_image = preprocess_for_train(decoded_images,image_size,image_size,None)

# 将处理后的图像和标签数据通过tf.train.shuffle_batch()整理成神经网络训练时需要的batch
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch,label_batch = tf.train.shuffle_batch([images,labels],
                                                 batch_size=batch_size,
                                                 capacity=capacity,
                                                 min_after_dequeue=min_after_dequeue)


# 训练模型
# 定义神经网络的结构以及优化过程。image_batch可以作为输入提供给神经网络的输入层
# label_batch则提供了输入batch中样例的正确答案

def inference(input_tensor,weights1,biases1,weights2,biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)
    return tf.matmul(layer1,weights2) + biases2
# 模型相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARAZION_RATE = 0.0001
TRAINING_STEPS = 50000

weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(image_batch, weights1, biases1, weights2, biases2)

# 计算交叉熵及其平均值
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=label_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# 损失函数计算
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZION_RATE)
regularazion = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularazion

# 优化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 初始化会话，开始训练过程。
with tf.Session() as sess:
    # tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # 训练神经网络
    for i in range(TRAINING_STEPS ):
        sess.run(train_step)
        if i % 1000 == 0:
            print("After %d training step(s), loss is %g " % (i, sess.run(loss)))
            print("After %d training step(s), loss is %g " % (i, sess.run(loss)))

    coord.request_stop()
    coord.join(threads)
