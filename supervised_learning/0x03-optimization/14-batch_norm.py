#!/usr/bin/env python3
""" Batch Normalization Upgraded """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """  creates a batch normalization layer for a neural network
    in tensorflow
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n, activation=None, kernel_initializer=init)

    m, s = tf.nn.moments(x(prev), axes=[0])

    beta = tf.Variable(tf.constant(0.0, shape=[n]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]),
                        name='gamma', trainable=True)

    f = (tf.nn.batch_normalization(x(prev), mean=m, variance=s,
         offset=beta, scale=gamma, variance_epsilon=1e-8))
    if activation is None:
        return f
    return activation(f)
