#!/usr/bin/env python3
"""Simple neuron class"""


import numpy as np


class Neuron:
    """Simple neuron class"""

    def __init__(self, nx):
        """nx: number of input features"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.ndarray((1, nx))
        self.__W[0] = np.random.normal(size=nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Return weights"""
        return self.__W

    @property
    def b(self):
        """Return bias"""
        return self.__b

    @property
    def A(self):
        """Return activation values"""
        return self.__A

    def forward_prop(self, X):
        """Update and return sigmoid activation values"""
        self.__A = 1 / (1 + np.exp(-1 * (np.dot(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        """Calculate cost using logistic regression"""
        costsum = 0
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
