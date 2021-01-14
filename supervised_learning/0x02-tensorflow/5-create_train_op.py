#!/usr/bin/env python3
""" Train_Op """
import tensorflow as tf


def create_train_op(loss, alpha):
    """ creates the training operation for the network """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
