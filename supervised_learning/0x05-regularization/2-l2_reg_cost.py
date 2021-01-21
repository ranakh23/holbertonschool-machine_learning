#!/usr/bin/env python3

"""cost"""
import tensorflow as tf


def l2_reg_cost(cost):
    """cost"""
    return cost + tf.losses.get_regularization_losses()
