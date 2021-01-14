#!/usr/bin/env python3
""" One-Hot Encode """
import numpy as np


def one_hot_encode(Y, classes):
    """ converts a numeric label vector into a one-hot matrix """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if type(classes) != int or classes <= np.amax(Y):
        return None
    one_hot = np.zeros((classes, len(Y)))
    axis = np.arange(len(Y))
    one_hot[Y, axis] = 1
    return one_hot
