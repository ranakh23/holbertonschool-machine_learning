#!/usr/bin/env python3
"""Implement add_matrices2D function"""


def add_matrices2D(mat1, mat2):
    """
        Add two matrices element-wise
        mat1: 2D matrix of ints/floats
        mat2: 2D matrix of ints/floats
        Return: Return new matrix or
                None if two matrices are not the same shape
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    sum_matrix = []
    for row1, row2 in zip(mat1, mat2):
        sum_matrix.append([])
        for x, y in zip(row1, row2):
            sum_matrix[-1].append(x + y)
    return sum_matrix
