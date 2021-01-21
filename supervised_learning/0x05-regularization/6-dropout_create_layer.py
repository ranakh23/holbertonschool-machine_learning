#!/usr/bin/env python3
""" Module to create a Layer with dropout """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ Create a layer with dropout """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer)
    layer_drop = tf.layers.Dropout(keep_prob)
    return layer_drop(layer(prev))
