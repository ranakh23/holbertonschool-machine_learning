#!/usr/bin/env python3
""" Module to evaluate early Stopping """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """  heads up gradient descent early stopping """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count != patience:
        return False, count
    return True, count
