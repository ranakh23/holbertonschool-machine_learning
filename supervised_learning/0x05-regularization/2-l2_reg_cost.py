#!/usr/bin/env python3
""" Tensorflow L2 Regularization Cost """
import tensorflow as tf


def l2_reg_cost(cost):
    """ Cost with L2 regularization """
    return cost + tf.losses.get_regularization_losses()
