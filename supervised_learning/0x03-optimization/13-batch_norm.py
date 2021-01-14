#!/usr/bin/env python3
""" Batch Normalization """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes an unactivated output of a
    neural network using batch normalization
    """

    m = np.mean(Z, axis=0)
    s = np.var(Z, axis=0)
    Z_norm = (Z - m) / ((s + epsilon)**(1/2))
    Z_tilde = gamma*Z_norm + beta
    return Z_tilde
