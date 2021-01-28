#!/usr/bin/env python3
""" Convolve images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Convolve images with channels"""
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    else:
        ph, pw = 0, 0
    pad_result = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)
    conv_h = (h - kh + 2 * ph) // sh + 1
    conv_w = (w - kw + 2 * pw) // sw + 1
    conv_result = np.zeros((m, conv_h, conv_w))

    for row in range(conv_h):
        for col in range(conv_w):
            conv_result[:, row, col] = np.sum(
                np.multiply(
                    pad_result[:, row*sh:row*sh + kh,
                               col*sw:col*sw + kw],
                    kernel
                ),
                axis=(1, 2, 3)
            )
    return conv_result
