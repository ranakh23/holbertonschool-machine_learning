#!/usr/bin/env python3
"""Densenet 151 architecture"""


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Return a Densenet 151 model"""
    indata = K.Input((224, 224, 3))
    out = K.layers.BatchNormalization()(indata)
    out = K.layers.Activation('relu')(out)
    out = K.layers.Conv2D(growth_rate * 2, 7, 2, padding='same',
                          kernel_initializer='he_normal')(out)
    out = K.layers.MaxPool2D(2)(out)
    out, nb_filters = dense_block(out, growth_rate * 2, growth_rate, 6)
    out, nb_filters = transition_layer(out, nb_filters, compression)
    out, nb_filters = dense_block(out, nb_filters, growth_rate, 12)
    out, nb_filters = transition_layer(out, nb_filters, compression)
    out, nb_filters = dense_block(out, nb_filters, growth_rate, 24)
    out, nb_filters = transition_layer(out, nb_filters, compression)
    out, nb_filters = dense_block(out, nb_filters, growth_rate, 16)
    out = K.layers.AvgPool2D(7)(out)
    out = K.layers.Dense(1000, kernel_initializer='he_normal',
                         activation='softmax')(out)
    return K.Model(indata, out)
