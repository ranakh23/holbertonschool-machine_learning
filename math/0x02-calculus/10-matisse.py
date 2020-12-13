#!/usr/bin/env python3
"""Implement poly_derivative(poly) function"""


def poly_derivative(poly):
    """
        Calculate the derivative of a polynomial represented by a list of
        coefficients
        poly: list of integers
        Return: list of integers, or None if poly is invalid
    """
    if not type(poly) is list or len(poly) == 0 or type(poly[0]) is not int:
        return None

    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)

    if derivative == []:
        derivative = [0]

    return derivative
