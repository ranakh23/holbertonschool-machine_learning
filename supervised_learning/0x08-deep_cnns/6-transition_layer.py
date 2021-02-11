#!/usr/bin/env python3
"""CNN transition layer"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """CNN transition layer"""
    out = K.layers.BatchNormalization()(X)
    out = K.layers.Activation('relu')(out)
    out = K.layers.Conv2D(int(nb_filters * compression), 1,
                          kernel_initializer='he_normal')(out)
    return K.layers.AvgPool2D(2)(out), int(nb_filters * compression)
