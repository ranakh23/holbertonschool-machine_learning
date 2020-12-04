#!/usr/bin/env python3
"""Implement cat_arrays function"""


def cat_arrays(arr1, arr2):
    """
        Concatenate two arrays
        arr1: list of ints/floats
        arr2: list of ints/floats
        Return: new list of ints/floats
    """
    new_list = list(arr1) + arr2
    return new_list
