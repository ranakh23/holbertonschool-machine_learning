#!/usr/bin/env python3
""" This module contains the function pool_backward. """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.
    """
    A_sh = A_prev.shape
    kh, kw = kernel_shape
    m, h_prev, w_prev, c_prev = A_sh
    sh, sw = stride
    _, h_new, w_new, c_new = dA.shape
    dA_prev = np.zeros(A_sh)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                x = j * sh
                y = k * sw
                for l in range(c_new):
                    currP = A_prev[i, x: x + kh, y: y + kw, l]
                    currZ = dA[i, j, k, l]
                    if mode == "max":
                        general = np.zeros(kernel_shape)
                        maxV = np.amax(currP)
                        np.place(general, currP == maxV, 1)
                        dA_prev[i, x: x + kh, y: y + kw, l] += general * currZ
                    else:
                        avg = currZ / (kh * kw)
                        general = np.ones(kernel_shape)
                        dA_prev[i, x: x + kh, y: y + kw, l] += general * avg
    return dA_prev
