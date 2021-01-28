#!/usr/bin/env python3
""" convolutions on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Same convolution on grayscale images"""
    kh, kw = kernel.shape
    m, imh, imw = images.shape
    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)
    padded = np.pad(images, ((0,), (ph,), (pw,)))
    ans = np.zeros((m, imh, imw))
    for i in range(imh):
        for j in range(imw):
            ans[:, i, j] = (padded[:, i: i + kh, j: j + kw] *
                            kernel).sum(axis=(1, 2))
    return ans
