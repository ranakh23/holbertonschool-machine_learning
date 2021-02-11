#!/usr/bin/env python3
"""Inception block builder"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Inception block builder"""
    onebyone = K.layers.Conv2D(filters[0], 1, activation='relu')(A_prev)
    threebythree = K.layers.Conv2D(filters[1], 1, activation='relu')(A_prev)
    threebythree = K.layers.Conv2D(filters[2], 3, padding='same',
                                   activation='relu')(threebythree)
    fivebyfive = K.layers.Conv2D(filters[3], 1, activation='relu')(A_prev)
    fivebyfive = K.layers.Conv2D(filters[4], 5, padding='same',
                                 activation='relu')(fivebyfive)
    pooling = K.layers.MaxPool2D(3, 1, padding='same')(A_prev)
    pooling = K.layers.Conv2D(filters[5], 1,
                              activation='relu')(pooling)
    return K.layers.concatenate([onebyone, threebythree, fivebyfive, pooling])
