#!/usr/bin/env python3
""" Convolutions with pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ Convolutions with pooling"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    conv_h = (h - kh) // sh + 1
    conv_w = (w - kw) // sw + 1
    conv_result = np.zeros((m, conv_h, conv_w, c))
    for row in range(conv_h):
        for col in range(conv_w):
            if mode == 'max':
                conv_result[:, row, col] = np.max(
                    images[:, row * sh:(kh + (row * sh)),
                           col * sw:(kw + (col * sw))], axis=(1, 2)
                )
            if mode == 'avg':
                conv_result[:, row, col] = np.mean(
                    images[:, row * sh:(kh + (row * sh)),
                           col * sw:(kw + (col * sw))],
                    axis=(1, 2)
                )
    return conv_result
