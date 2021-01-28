#!/usr/bin/env python3
""" Same convolution on grayscale images """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ Same convolution with padding """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h, pad_w = padding
    conv_h = h + 2 * pad_h - kh + 1
    conv_w = w + 2 * pad_w - kw + 1
    pad_img = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                     mode='constant', constant_values=0)
    conv_result = np.zeros((m, conv_h, conv_w))

    for row in range(conv_h):
        for col in range(conv_w):
            conv = np.sum(pad_img[:, row:row + kh, col:col + kw] * kernel,
                          axis=(1, 2))
            conv_result[:, row, col] = conv
    return conv_result
