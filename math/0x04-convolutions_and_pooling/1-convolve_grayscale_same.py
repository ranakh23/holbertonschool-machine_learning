#!/usr/bin/env python3
""" convolutions on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    if kh % 2 == 0:
        pad_h = kh // 2
    if kw % 2 == 0:
        pad_w = kw / 2
    new_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant', constant_values=0)
    conv_result = np.zeros((m, h, w))
    img = np.arange(m)
    for row in range(h):
        for col in range(w):
            conv_result[img, row, col] = np.sum(np.multiply(new_images[img,
                                                            row:row + kh,
                                                            col:col + kw],
                                                            kernel),
                                                axis=(1, 2))
    return conv_result
