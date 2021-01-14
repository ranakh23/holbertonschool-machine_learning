#!/usr/bin/env python3
""" Accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction """
    pred = tf.argmax(y_pred, 1)
    val = tf.argmax(y, 1)
    equality = tf.equal(pred, val)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy
