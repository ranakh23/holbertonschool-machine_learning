#!/usr/bin/env python3
"""Implement a modified LeNet-5 architecture using Keras"""


import tensorflow.keras as K


def lenet5(X):
    """Implement a modified LeNet-5 architecture using Keras"""
    out = K.layers.Conv2D(6, 5, padding='same',
                          activation='relu',
                          kernel_initializer='he_normal')(X)
    out = K.layers.MaxPooling2D(2, 2)(out)
    out = K.layers.Conv2D(16, 5, padding='valid', activation='relu',
                          kernel_initializer='he_normal')(out)
    out = K.layers.MaxPooling2D(2, 2)(out)
    out = K.layers.Flatten()(out)
    out = K.layers.Dense(120, activation='relu',
                         kernel_initializer='he_normal')(out)
    out = K.layers.Dense(84, activation='relu',
                         kernel_initializer='he_normal')(out)
    out = K.layers.Dense(10, activation='softmax',
                         kernel_initializer='he_normal')(out)
    model = K.Model(X, out)
    model.compile('Adam', metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model
