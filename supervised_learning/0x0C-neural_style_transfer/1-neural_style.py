#!/usr/bin/env python3
"""
Neural style transfer class
"""

import tensorflow as tf
import numpy as np


class NST:
    """
    A class including model designed to do style transfer between two images
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize the class.
        @style_image: image for style to mimic
        @content_image: image we're modifying to match style
        @alpha: learning rate for content
        @beta: learning rate for style
        """
        tf.enable_eager_execution()
        if ((not isinstance(style_image, np.ndarray)
             or len(style_image.shape) != 3 or style_image.shape[2] != 3)):
            raise TypeError("style_image must be a numpy.ndarray "
                            "with shape (h, w, 3)")
        if ((not isinstance(content_image, np.ndarray)
             or len(content_image.shape) != 3 or content_image.shape[2] != 3)):
            raise TypeError("content_image must be a "
                            "numpy.ndarray with shape (h, w, 3)")
        if ((not isinstance(alpha, (float, int, complex))
             or alpha < 0)):
            raise TypeError("alpha must be a non-negative number")
        if ((not isinstance(beta, (float, int, complex))
             or beta < 0)):
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Scales an image to be between 0-1 and resizes max dim to 512
        """
        max_side = max(image.shape)
        dims = [int(image.shape[0] / max_side * 512),
                int(image.shape[1] / max_side * 512)]
        image = tf.compat.v1.image.resize_bicubic(image[np.newaxis, ...], dims)
        image -= tf.reduce_min(image)
        return image / tf.reduce_max(image)

    def load_model(self):
        """
        Loads VGG19 keras model
        """
        base_model = tf.keras.applications.vgg19.VGG19(include_top=False)
        base_model.trainable = False
        layers = self.style_layers + [self.content_layer]
        model = tf.keras.Model(inputs=base_model.input,
                               outputs=[base_model.get_layer(layer).output
                                        for layer in layers])
        self.model = model
        return self.model
