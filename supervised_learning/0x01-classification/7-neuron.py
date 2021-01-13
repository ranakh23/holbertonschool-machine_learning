#!/usr/bin/env python3
"""Neuron class"""


import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """Neuron class"""

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
        """
        self.__A = np.zeros((1, X.shape[1]))
        for ex in range(X.shape[1]):
            expsum = 0
            for feat in range(X.shape[0]):
                expsum += self.__W[0][feat] * X[feat][ex] + self.__b
            self.__A[0][ex] = 1 / (1 + np.exp(-1 * expsum))
        """
        self.__A = 1 / (1 + np.exp(-1 * (np.dot(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        """Calculate cost using logistic regression"""
        costsum = 0
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()

    def evaluate(self, X, Y):
        """Evaluate neuron's predictions"""
        self.__A = np.ndarray((1, X.shape[1]))
        self.__A[0] = self.forward_prop(X)
        return np.round(self.__A).astype(int), self.cost(Y, self.__A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Update neuron's weights and bias"""
        self.__W[0] = (self.__W[0] - alpha *
                       np.dot(X, (A - Y).T).T[0] / X.shape[1])
        self.__b -= alpha * (A[0] - Y[0]).mean()

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
            A = self.forward_prop(X)
            if verbose and not (itrcount % step):
                print("Cost after {} iterations: {}"
                      .format(itrcount, self.cost(Y, A)))
                if graph:
                    losses.append(self.cost(Y, A))
                    graphx.append(itrcount)
            self.gradient_descent(X, Y, A, alpha)
            itrcount += 1
        self.forward_prop(X)
        if verbose:
            print("Cost after {} iterations: {}"
                  .format(itrcount, self.cost(Y, self.__A)))
        if graph:
            plt.plot(graphx, losses, "b-")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
        return self.evaluate(X, Y)
