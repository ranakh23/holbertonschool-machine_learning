#!/usr/bin/env python3
"""ResNet50 network"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Return a resnet50 model"""
    indata = K.layers.Input((224, 224, 3))
    out = K.layers.Conv2D(64, 7, 2, padding='same',
                          kernel_initializer='he_normal')(indata)
    out = K.layers.BatchNormalization()(out)
    out = K.layers.Activation('relu')(out)
    out = K.layers.MaxPool2D(3, 2, padding='same')(out)
    filters = [64, 64, 256]
    out = projection_block(out, filters, 1)
    out = identity_block(out, filters)
    out = identity_block(out, filters)
    filters = [128, 128, 512]
    out = projection_block(out, filters, 2)
    out = identity_block(out, filters)
    out = identity_block(out, filters)
    out = identity_block(out, filters)
    filters = [256, 256, 1024]
    out = projection_block(out, filters, 2)
    out = identity_block(out, filters)
    out = identity_block(out, filters)
    out = identity_block(out, filters)
    out = identity_block(out, filters)
    out = identity_block(out, filters)
    filters = [512, 512, 2048]
    out = projection_block(out, filters, 2)
    out = identity_block(out, filters)
    out = identity_block(out, filters)
    out = K.layers.AvgPool2D(7)(out)
    out = K.layers.Dense(1000, kernel_initializer='he_normal',
                         activation='softmax')(out)
    return K.Model(indata, out)
