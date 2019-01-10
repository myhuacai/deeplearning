#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt

HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIMESTEPS = 10
TRAINING_STEPS = 10000
TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
BATCH_SIZE = 32
SAMPLE_GAP = 0.01  # 采样间隔


def generate_data(seq):
    X = []
    y = []

    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i:i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(X, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
                                        for _ in range(NUM_LAYERS)])
    out_puts, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = out_puts[:, -1, :]
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    if not is_training:
        return predictions, None, None
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(),
                                               optimizer='Adagrad', learning_rate=0.1)
    return predictions, loss, train_op


def train(sess, train_X, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope('model1'):
        predictions, loss, train_op = lstm_model(X, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_EXAMPLES):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print("train steps: " + str(i) + ",loss: " + str(l))


def run_eval(sess, test_X, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope('model1', reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Mean Square Error is : %f" % rmse)

    plt.figure()
    plt.plot(predictions, label='predictions', linewidth=5)
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()
    # print("--------")


test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP

train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

test_X, test_y = generate_data(
    np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
# print(test_X)

with tf.Session() as sess:
    train(sess, train_X, train_y)
    run_eval(sess, test_X, test_y)
