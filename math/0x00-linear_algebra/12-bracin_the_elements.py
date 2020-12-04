#!/usr/bin/env python3
"""Implement np_elementwise(mat1, mat2) function"""


def np_elementwise(mat1, mat2):
    """
        Calculates element-wise addition, subtraction, multiplication,
            and division
        mat1: numpy.ndarray
        mat2: numpy.ndarray
        Return: tuple containing element-wise sum, difference,
            product, and quotient
    """
    import numpy as np
    return (np.add(mat1, mat2), np.subtract(mat1, mat2),
            np.multiply(mat1, mat2), np.divide(mat1, mat2))
