#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data


INPUT_NOCE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model.ckpt"


# import mnist_inference
# import mnist_train
EVAL_INTERVAL_SECS = 10

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NOCE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    return layer2


def evaluate(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NOCE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    validate_feed = {x: mnist.train.images, y_: mnist.train.labels}

    y = inference(x, None)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variable_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variable_to_restore)


    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and  ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            # saver.restore(sess, ckpt.model_checkpoit_path.split('/'[-1].split('-')[-1]))
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
            print("After %s training steps,validation accuracy =  %g." % (global_step, accuracy_score))
        else:
            print("NO checkpoit file found")
            return
        sess.run()
    # time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("mnist", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()


