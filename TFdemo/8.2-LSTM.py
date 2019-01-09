#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf

batch_size = 5
lstm_hidden_size = 10
# 定义一个LSTM结构，在TensorFlow中通过一句简单的命令就可以实现一个完整的LSTM结构
# LSTM中使用的变量也会在改函数中自动被声明
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

# 将LSTM中的状态初始化为全0数组。BasicLSTMCell类提供了zero_state函数来生成全零的初始状态。
# state是一个包含两个张量的LSTMStateTupe类，其中state.c和state.h分别对应了c的状态和h的状态
# 和其他网络类似，在优化循环神经网络时，每次也会使用一个batch的训练样本
state = lstm.zero_state(batch_size=batch_size,dtype=tf.float32)

# 定义损失函数
loss = 0.0
num_steps =10
current_input = None
def fully_connented(lstm_output):
    pass
def calc_loss(filnal_output,expected_output):
    pass

# 虽然在测试时循环神经网络可以处理任意长度的序列，但是在训练中为了将循环网络展开成前馈神经网络，我们需要知道训练数据的序列长度。
# 在以下代码中，用num_steps来表示这个长度
for i in range(num_steps):
    # 在第一个时刻声明LSTM结构中使用的变量，在之后的时候都需要复用之前定义好的变量
    if i >0 :
        tf.get_variable_scope().reuse_variables()
    # 每一步处理实践序列中的一个时刻。将当前输入current_input和前一时刻状态state传入定义的LSTM结构可以
    # 得到当前LSTM的输出lstn_output和更新状态state。
    # lstm_output用于输出给其他层，state用于输出给下一时刻，它们在的肉蒲团等方面可以有不同的处理方式
    lstm_output,state = lstm(current_input,state)
# 将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出
    filnal_output = fully_connented(lstm_output)
    expected_output = None
    # 计算当前时刻的损失
    loss += calc_loss(filnal_output,expected_output)

# 使用一般流程进行训练--略