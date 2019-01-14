#!/usr/bin/env python
# -*- coding:utf-8 -*-

import  matplotlib.pyplot as plt
import tensorflow as tf

# 读取图片
image_data = tf.gfile.FastGFile("data/dog3.jpg", 'br').read()
#
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_data)
    plt.imshow(img_data.eval())
    plt.show()

    # 将图片数据转换成实数类型
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # # 图像上下翻转
    # fliped1 = tf.image.flip_up_down(img_data)
    # plt.imshow(fliped1.eval())
    # 左右翻转
    fliped = tf.image.flip_left_right(img_data)
    plt.imshow(fliped.eval())
    # 沿着对角线翻转
    # transposed = tf.image.transpose_image(img_data)
    # plt.imshow(transposed.eval())
    # # 以50%概率上下翻转
    # fliped = tf.image.random_flip_up_down(img_data)
    # plt.imshow(fliped.eval())
    # # 以50%概率左右翻转
    # fliped = tf.image.random_flip_left_right(img_data)
    # plt.imshow(fliped.eval())
    plt.show()
# 1.图像编码处理
# 读取图片
# image_data = tf.gfile.FastGFile("data/dog1.jpg",'br').read()

# # # print(image_data)
# with tf.Session() as sess:
#     # 解码原图片并展示出来
#     img_data = tf.image.decode_jpeg(image_data)
#     plt.imshow(img_data.eval())
#     plt.show()
#     encoded_image = tf.image.encode_jpeg(img_data)
#     # print(sess.run(encoded_image))
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
#     # 将图片数据转换成实数类型
#     img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
#     # 重新转换成300X900的图片，并展示出调整后的图片
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
# with tf.Session() as sess:
    # img_data = tf.image.decode_jpeg(image_data)
    # img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # fliped1 = tf.image.flip_up_down(img_data)
    # plt.imshow(fliped1.eval())
    # fliped = tf.image.flip_left_right(img_data)
    # plt.imshow(fliped.eval())
    # transposed = tf.image.transpose_image(img_data)
    # plt.imshow(transposed.eval())
    # fliped = tf.image.random_flip_up_down(img_data)
    # plt.imshow(fliped.eval())
    # fliped = tf.image.random_flip_left_right(img_data)
    # plt.imshow(fliped.eval())
    # plt.show()

# 4.图像色彩调整
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_data)
#     img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
#     # 亮度
#     # adjusted = tf.image.adjust_brightness(img_data,0.5)
#     # adjusted = tf.image.adjust_brightness(img_data, -0.5)
#     # adjusted = tf.clip_by_value(adjusted,0.0,1.0)
#     # adjusted = tf.image.random_brightness(img_data,0.5)
#     #
#     # 对比度
#     # adjusted = tf.image.adjust_contrast(img_data,0.5)
#     # adjusted = tf.image.adjust_contrast(img_data, 5)
#     # adjusted = tf.image.random_contrast(img_data,0.1,5)
#     # 色相
#     # adjusted = tf.image.adjust_hue(img_data,0.3)
#     # adjusted = tf.image.adjust_hue(img_data, 0.1)
#     # adjusted = tf.image.adjust_hue(img_data, 0.9)
#     # adjusted = tf.image.adjust_hue(img_data, 0.6)
#     # adjusted = tf.image.random_hue(img_data,0.5)
#     # 饱和度
#     # adjusted = tf.image.adjust_saturation(img_data,-5)
#     # adjusted = tf.image.adjust_saturation(img_data, 5)
#     # adjusted = tf.image.random_saturation(img_data,2,5)
#     # 图像标准化 均值为0 方差变为1
#     adjusted = tf.image.per_image_standardization(img_data)
#     plt.imshow(adjusted.eval())
#     plt.show()

# 5.处理标注窗
#
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(image_data)
#     img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
#     # 将图片缩小一些，这样可视化能让标注框更加清楚
#     img_data = tf.image.resize_images(img_data,[180,267],method=1)
#     batched = tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
#     boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
#     # result = tf.image.draw_bounding_boxes(batched,boxes=boxes)
#     # plt.imshow(result[0].eval())
#     # print(result)
#     # 随机截取图片
#     begin,size,bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(img_data),
#                                                                       bounding_boxes=boxes,
#                                                                       min_object_covered=0.4)
#     batched = tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
#     image_with_box = tf.image.draw_bounding_boxes(batched,bbox_for_draw)
#     distored_image = tf.slice(img_data,begin,size=size)
#     plt.imshow(distored_image.eval())
#     plt.show()
