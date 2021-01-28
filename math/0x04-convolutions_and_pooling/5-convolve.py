#!/usr/bin/env python3
""" Convolutions with different kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ Convolve with different kernels"""
    m, h, w, c = images.shape
    kh, kw, kc, channels = kernels.shape
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
    conv_result = np.zeros((m, conv_h, conv_w, channels))

    for row in range(conv_h):
        for col in range(conv_w):
            for channel in range(channels):
                conv_result[:, row, col, channel] = np.sum(
                    pad_result[:, row * sh:(kh + (row * sh)),
                               col * sw:(kw + (col * sw))]
                    * kernels[:, :, :, channel],
                    axis=(1, 2, 3)
                )
    return conv_result
