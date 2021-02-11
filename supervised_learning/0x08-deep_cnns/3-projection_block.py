#!/usr/bin/env python3
"""Identity block maker"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Return an identity block"""
    out = K.layers.Conv2D(filters[0], 1, s,
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
    out2 = K.layers.Conv2D(filters[2], 1, s,
                           kernel_initializer='he_normal')(A_prev)
    out2 = K.layers.BatchNormalization()(out2)
    out = K.layers.add([out, out2])
    return K.layers.Activation('relu')(out)
