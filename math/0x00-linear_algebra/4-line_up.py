#!/usr/bin/env python3
"""Implement add_arrays function"""


def add_arrays(arr1, arr2):
    """
        Add two arrays element-wise
        arr1: list of ints/floats
        arr2: list of ints/floats
        Return: list of ints/floats or None if arrays are not same shape
    """
    if len(arr1) != len(arr2):
        return None

    sum_matrix = []
    for x, y in zip(arr1, arr2):
        sum_matrix.append(x + y)
    return sum_matrix
