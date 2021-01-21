#!/usr/bin/env python3
""" Forward Prop with Dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Forward propagation using Dropout """
    cache = {}
    cache["A0"] = X
    for layer in range(L):
        A = "A{}".format(layer)
        A_next = "A{}".format(layer + 1)
        W = "W{}".format(layer + 1)
        b = "b{}".format(layer + 1)
        D_next = "D{}".format(layer + 1)
        A_layer = np.matmul(weights[W], cache[A]) + weights[b]
        dropout = np.random.binomial(1, keep_prob, size=A_layer.shape)
        if layer == L - 1:
            cache[A_next] = np.exp(A_layer)/np.sum(np.exp(A_layer),
                                                   axis=0,
                                                   keepdims=True)
        else:
            cache[A_next] = np.tanh(A_layer)
            cache[D_next] = dropout
            cache[A_next] *= dropout
            cache[A_next] /= keep_prob
    return cache
