#!/usr/bin/env python3

"""Creates a Tensorflow layer that includes L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a Tensorflow layer that includes L2 regularization
        - prev is a tensor containing the output of the previous layer
        - n is the number of nodes the new layer should contain
        - activation is the activation function that should be used
         on the layer
        - lambtha is the L2 regularization parameter
        Returns: the output of the new layer
    """
    regul = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regul)

    return model(prev)
