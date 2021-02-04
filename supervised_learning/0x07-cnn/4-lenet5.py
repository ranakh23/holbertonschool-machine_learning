#!/usr/bin/env python3
""" function lenet5 """
import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow.
    """
    He = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu, kernel_initializer=He)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=He)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    pool2 = tf.layers.Flatten()(pool2)
    dense1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                             kernel_initializer=He)(pool2)
    dense2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                             kernel_initializer=He)(dense1)
    dense3 = tf.layers.Dense(units=10, kernel_initializer=He)(dense2)
    y_pred = tf.nn.softmax(dense3)
    y_pred_tag = tf.argmax(dense3, 1)
    y_tag = tf.argmax(y, 1)
    comp = tf.equal(y_pred_tag, y_tag)
    acc = tf.reduce_mean(tf.cast(comp, tf.float32))
    loss = tf.losses.softmax_cross_entropy(y, dense3)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    return y_pred, train_op, loss, acc
