#!/usr/bin/env python
# -*- coding:utf-8 -*-

import  matplotlib.pyplot as plt
import tensorflow as tf


# 1.图像编码处理
image_data = tf.gfile.FastGFile("data/05.jpg",'br').read()

# print(image_data)
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_data)
#     # print(img_data.eval())
#     plt.imshow(img_data.eval())
#     plt.show()
#     encoded_image = tf.image.encode_jpeg(img_data)
#     print(sess.run(encoded_image))
#     with tf.gfile.GFile("data/output","wb") as f:
#         f.write(encoded_image.eval())

# image_data = tf.gfile.GFile("data/output",'br').read()
# print(image_data)
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_data)
#     # print(img_data.eval())
#     plt.imshow(img_data.eval())
#     plt.show()
#

# 2.图像大小调整

# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_data)
#     img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
#     # print(img_data.eval())
#     resized = tf.image.resize_images(img_data,[300,900],method=1)
#     plt.imshow(resized.eval())
#     plt.show()
#1920X1200
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_data)
#     img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
#     # resized = tf.image.resize_images(img_data, [300, 900], method=1)
#     # plt.imshow(resized.eval())
#     # croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
#     # plt.imshow(croped.eval())
#     # padded = tf.image.resize_image_with_crop_or_pad(img_data,3000,3000)
#     # plt.imshow(padded.eval())
#     central_croped = tf.image.central_crop(img_data,0.8)
#     plt.imshow(central_croped.eval())
#     plt.show()

# 3.图像翻转
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # fliped1 = tf.image.flip_up_down(img_data)
    # plt.imshow(fliped1.eval())
    # fliped = tf.image.flip_left_right(img_data)
    # plt.imshow(fliped.eval())
    # transposed = tf.image.transpose_image(img_data)
    # plt.imshow(transposed.eval())
    # fliped = tf.image.random_flip_up_down(img_data)
    # plt.imshow(fliped.eval())
    fliped = tf.image.random_flip_left_right(img_data)
    plt.imshow(fliped.eval())
    plt.show()

# 4.图像色彩调整
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    fliped = tf.image.random_hue()
    plt.imshow(fliped.eval())
    plt.show()
# 5.处理标注窗