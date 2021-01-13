#!/usr/bin/env python3
"""Simple neuron class"""


import numpy as np


class Neuron:
    """Neuron class"""

    def __init__(self, nx):
        """nx: number of input features"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.ndarray((1, nx))
        self.W[0] = np.random.normal(size=nx)
        self.b = 0
        self.A = 0
