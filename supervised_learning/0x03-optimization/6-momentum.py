#!/usr/bin/env python3
""" Momentum Upgraded """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ training for a NN in tensorflow using GD - momentum opt algorithm """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
