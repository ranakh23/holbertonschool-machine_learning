#!/usr/bin/env python3
""" Layers """
import tensorflow as tf


def create_layer(prev, n, activation):
    """ creates layers """
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    tensor_l = (tf.layers.Dense(units=n, activation=activation,
                kernel_initializer=initialize, name="layer"))
    return tensor_l(prev)
