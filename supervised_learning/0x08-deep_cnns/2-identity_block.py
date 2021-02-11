#!/usr/bin/env python3
"""Create an identity block maker"""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Return an identity block"""
    out = K.layers.Conv2D(filters[0], 1,
                          kernel_initializer='he_normal')(A_prev)
    out = K.layers.BatchNormalization()(out)
    out = K.layers.Activation('relu')(out)
    out = K.layers.Conv2D(filters[1], 3, padding='same',
                          kernel_initializer='he_normal')(out)
    out = K.layers.BatchNormalization()(out)
    out = K.layers.Activation('relu')(out)
    out = K.layers.Conv2D(filters[2], 1,
                          kernel_initializer='he_normal')(out)
    out = K.layers.BatchNormalization()(out)
    out = K.layers.add([out, A_prev])
    return K.layers.Activation('relu')(out)
