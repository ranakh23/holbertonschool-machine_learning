#!/usr/bin/env python3
""" Shuffle Data """
import numpy as np


def shuffle_data(X, Y):
    """  shuffles the data points in two matrices the same way """
    st0 = np.random.get_state()
    X_shuffled = np.random.permutation(X)
    np.random.set_state(st0)
    Y_shuffled = np.random.permutation(Y)
    return X_shuffled, Y_shuffled
