#!/usr/bin/env python3
"""Calculate precision of a confusion matrix"""


import numpy as np


def precision(confusion):
    """Calculate precision of a confusion matrix"""
    return np.asarray([confusion[row][row] / confusion[:, row].sum()
                       for row in range(confusion.shape[0])])
