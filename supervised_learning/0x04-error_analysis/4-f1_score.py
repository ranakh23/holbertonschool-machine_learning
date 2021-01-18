#!/usr/bin/env python3
"""Calculate f1 score of a confusion matrix"""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculate f1 score of a confusion matrix"""
    sens = sensitivity(confusion)
    prec = precision(confusion)
    return 2 * sens * prec / (sens + prec)
