#!/usr/bin/env python3
"""Train a keras model"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Train a keras model"""
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       shuffle=shuffle, verbose=verbose)
