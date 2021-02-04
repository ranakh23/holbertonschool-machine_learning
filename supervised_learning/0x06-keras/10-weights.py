#!/usr/bin/env python3
"""Save and load a keras model"""


import tensorflow.keras as K


def save_weights(network, filename, format='h5'):
    """Save a keras model"""
    network.save_weights(filename, format)


def load_weights(network, filename):
    """Load a whole keras model"""
    return network.load_weights(filename)
