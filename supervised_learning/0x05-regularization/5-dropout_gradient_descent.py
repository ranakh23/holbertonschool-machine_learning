#!/usr/bin/env python3
""" Gradient Descent with Dropout """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ Gradient descent with dropout regularization """
    weights_copy = weights.copy()
    m = Y.shape[1]
    for layer in range(L, 0, -1):
        A = "A{}".format(layer)
        A_prev = "A{}".format(layer - 1)
        W = "W{}".format(layer)
        W_next = "W{}".format(layer + 1)
        b = "b{}".format(layer)
        D = "D{}".format(layer)
        if layer == L:
            dz = cache[A] - Y
            dw = (np.matmul(cache[A_prev], dz.T) / m).T
        else:
            d1 = np.matmul(weights_copy[W_next].T, dz_prev)
            d2 = 1 - cache[A]**2
            dz = d1 * d2
            dz *= cache[D]
            dz /= keep_prob
            dw = np.matmul(dz, cache[A_prev].T) / m
        dz_prev = dz
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights[W] = weights_copy[W] - alpha * dw
        weights[b] = weights_copy[b] - alpha * db
