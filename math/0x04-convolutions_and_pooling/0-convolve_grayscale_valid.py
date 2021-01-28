#!/usr/bin/env python3
""" convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ Convolve on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    conv_h = h - kh + 1
    conv_w = w - kw + 1
    conv_result = np.zeros((m, conv_h, conv_w))
    img = np.arange(m)
    for i in range(conv_h):
        for j in range(conv_w):
            conv_result[img, i, j] = np.sum(np.multiply(
                images[img, i:i + kh, j:j + kw], kernel), axis=(1, 2))
    return conv_result
