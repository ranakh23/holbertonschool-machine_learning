#!/usr/bin/env python3
"""Create a confusion matrix"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Create a confusion matrix.
    labels: correct labels in one-hot form
    logits: predictions
    """
    return np.dot(labels.T, logits)
