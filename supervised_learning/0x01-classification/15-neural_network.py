#!/usr/bin/env python3
"""Simple neural network class"""


import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """
        Evaluate the neural network.
        X: Input data
        Y: Correct labels
        """
        return (self.forward_prop(X)[1].round().astype(int),
                self.cost(Y, self.__A2))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Perform a gradient descent step on neural network
        X: Input data
        Y: Correct labels
        A1: hidden layer activations
        A2: output layer activations
        alpha: learning rate
        """
        dz2 = A2 - Y
        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        self.__W2 -= alpha * np.dot(dz2, A1.T) / A1.shape[1]
        self.__b2 -= alpha * dz2.mean(axis=1, keepdims=True)
        self.__W1 -= alpha * np.dot(dz1, X.T) / X.shape[1]
        self.__b1 -= alpha * dz1.mean(axis=1, keepdims=True)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Train the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1:
                raise ValueError("step must be a positive integer")
        itrcount = 0
        losses = []
        graphx = []
        while itrcount < iterations:
            A1, A2 = self.forward_prop(X)
            if verbose and not (itrcount % step):
                print("Cost after {} iterations: {}"
                      .format(itrcount, self.cost(Y, A2)))
            if graph:
                losses.append(self.cost(Y, A2))
                graphx.append(itrcount)
            self.gradient_descent(X, Y, A1, A2, alpha)
            itrcount += 1
        self.forward_prop(X)
        if verbose:
            print("Cost after {} iterations: {}"
                  .format(itrcount, self.cost(Y, self.__A2)))
        if graph:
            plt.plot(graphx, losses, "b-")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
        return self.evaluate(X, Y)
