#!/usr/bin/env python3
"""Simple neural network class"""


import numpy as np


class NeuralNetwork:
    """Simple neural network class"""

    def __init__(self, nx, nodes):
        """
        nx: number of input features
        nodes: number of hidden layer nodes
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.ndarray((nodes, nx))
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__W2 = np.ndarray((1, nodes))
        self.__W2[0] = np.random.normal(size=nodes)
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def W2(self):
        return self.__W2

    @property
    def b1(self):
        return self.__b1

    @property
    def b2(self):
        return self.__b2

    @property
    def A1(self):
        return self.__A1

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculate forward propagation for neural network
        X: input data
        """
        self.__A1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-1 * self.__A1))
        self.__A2 = (np.dot(self.__W2, self.__A1) +
                     self.__b2)
        self.__A2 = 1 / (1 + np.exp(-1 * self.__A2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculate cost of the neural network
        Y: Correct labels
        A: Activation predictions.
        """
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
