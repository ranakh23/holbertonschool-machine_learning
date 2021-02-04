#!/usr/bin/env python3
"""Neural network with dropout"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Neural network with dropout"""
    reg = K.regularizers.l2
    model = K.Sequential()
    model.add(K.layers.Dense(layers[0], input_shape=(nx,),
                             activation=activations[0],
                             kernel_regularizer=reg(lambtha)))
    for layer, act in zip(layers[1:], activations[1:]):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layer, activation=act,
                                 kernel_regularizer=reg(lambtha)))
    return model
