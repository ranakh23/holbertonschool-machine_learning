#!/usr/bin/env python3
""" This module contains the function conv_backward. """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.
    """
    A_sh = A_prev.shape
    W_sh = W.shape
    m, h_prev, w_prev, c_prev = A_sh
    kh, kw, _, c_new = W_sh
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape
    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2) + 1
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2) + 1
    else:
        ph = pw = 0
    padded = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), mode='constant',
                    constant_values=0)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    dA = np.zeros(padded.shape)
    dW = np.zeros(W_sh)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                x = j * sh
                y = k * sw
                for l in range(c_new):
                    currZ = dZ[i, j, k, l]
                    currP = padded[i, x: x + kh, y: y + kw, :]
                    dA[i, x: x + kh, y: y + kw, :] += currZ * W[:, :, :, l]
                    dW[:, :, :, l] += currP * currZ
    if padding == 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]
    return dA, dW, db
