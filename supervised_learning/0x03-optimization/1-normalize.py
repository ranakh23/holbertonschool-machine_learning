#!/usr/bin/env python3
""" Normalize """
import numpy as np


def normalize(X, m, s):
    """  normalizes (standardizes) a matrix """
    z = (X - m)/s
    return z
