#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist = input_data.read_data_sets("./mnist",one_hot=True)

# batch size
batch = 100

# n batch
n_batch = mnist.train.num_examples // batch

# define
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001,dtype=tf.float32)

# define net
W1 = tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
# Wx_plus_b1 = tf.matmul(x,W1) + b1
L1 = tf.nn.tanh(tf.matmul(x,W1) + b1)
L1_drop = tf.nn.dropout(L1,keep_prob=keep_prob)


W2 = tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2 = tf.Variable(tf.zeros(300) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2) + b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([300,10],stddev = 0.1))
b3 = tf.Variable(tf.zeros(10) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop,W3) + b3)


# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

# train
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# initializer
init = tf.global_variables_initializer()

# correct_prediction
correct_prediction  = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

# accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# train

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        sess.run(tf.assign(lr,0.001*(0.95 ** epoch)))
        for batch in range(n_batch):
            xs_batch,ys_batch = mnist.train.next_batch(batch)
            sess.run(train_step,feed_dict={x:xs_batch,y:ys_batch,keep_prob:1.0})
        learning_rate = sess.run(lr)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc) + ", Learning Rate= " + str(learning_rate))








