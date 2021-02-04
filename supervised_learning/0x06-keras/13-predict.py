#!/usr/bin/env python3
"""Predict with a keras network"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Predict with a keras network"""
    return network.predict(data, verbose=verbose)
