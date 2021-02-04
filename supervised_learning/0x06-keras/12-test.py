#!/usr/bin/env python3
"""Evaluate a keras network"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Evaluate a keras network"""
    return network.evaluate(data, labels, verbose=verbose)
