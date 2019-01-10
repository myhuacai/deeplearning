#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import tensorflow as tf

TRAIN_DATA = "./data/ptb/ptb.train"
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEPS = 35


# 从文件中读取数据，并返回包含单词编号的数组
def read_data(file_path):
    with open(file_path, "r") as fin:
        # 将整个文档读进一个长字符串
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    # 转换为编号
    id_list = [int(w) for w in id_string.split()]
    return id_list


def make_batches(id_list, batch_size, num_steps):
    # 计算总的batch数量。每个batch包含的单词数量是batch_size*num_steps
    num_batches = (len(id_list) - 1) // (batch_size * num_steps)
    # 将数据整理成一个维度为[batch_size,num_batches*num_steps]的二维数组
    data = np.array(id_list[:num_batches * batch_size * num_steps])
    data = np.reshape(data, [batch_size, num_batches * num_steps])
    # 沿着第二个维度(列)，将数据切分成num_batches个batch，存入一个数组
    data_batches = np.split(data, num_batches, axis=1)
    # print(data_batches[0][1])
    # 重复上述操作，但是每个位置向右移动一位。这里得到的是RNN每一步输出所需要预测的下一个单词
    label = np.array(id_list[1:num_batches * batch_size * num_steps + 1])
    label = np.reshape(label, [batch_size, num_batches * num_steps])
    # print(label)
    label_batches = np.split(label, num_batches, axis=1)
    # print(label_batches[0][0])
    # print(list(zip(data_batches, label_batches))[0][0][0])
    return list(zip(data_batches, label_batches))


def main():
    train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEPS)
    # 这里添加训练模型代码
    print("over...")


if __name__ == "__main__":
    main()
