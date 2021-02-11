#!/usr/bin/env python3
"""Dense CNN block"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Return a dense block"""
    prev = X
    for layer in range(layers):
        cur = K.layers.BatchNormalization()(prev)
        cur = K.layers.Activation('relu')(cur)
        cur = K.layers.Conv2D(growth_rate * 4, 1,
                              kernel_initializer='he_normal')(cur)
        cur = K.layers.BatchNormalization()(cur)
        cur = K.layers.Activation('relu')(cur)
        cur = K.layers.Conv2D(growth_rate, 3, padding='same',
                              kernel_initializer='he_normal')(cur)
        prev = K.layers.concatenate([prev, cur])
        nb_filters += growth_rate
    return prev, nb_filters
