#!/usr/bin/env python3

"""Create a class NST that performs tasks for neural style transfer"""

import tensorflow as tf
import numpy as np


class NST:
    """class NST
    Public class attributes:
    style_layers = ['block1_conv1','block2_conv1', 'block3_conv1',
    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'"""

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class constructor
        style_image - the image used as a style reference,
        stored as a numpy.ndarray
        content_image - the image used as a content reference,
        stored as a numpy.ndarray
        alpha - the weight for content cost
        beta - the weight for style cost"""

        message = "style_image must be a numpy.ndarray with shape (h, w, 3)"

        if (not isinstance(style_image, np.ndarray) or
           len(style_image.shape) != 3 or style_image.shape[2] != 3):
            raise TypeError(message)

        message = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if (not isinstance(content_image, np.ndarray) or
           len(content_image.shape) != 3 or content_image.shape[2] != 3):
            raise TypeError(message)

        if not isinstance(alpha, (int, float, complex)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float, complex)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        image - a numpy.ndarray of shape (h, w, 3) containing
        the image to be scaled
        Returns: the scaled image"""

        message = "image must be a numpy.ndarray with shape (h, w, 3)"
        if (not isinstance(image, np.ndarray) or len(image.shape) != 3 or
           image.shape[2] != 3):
            raise TypeError(message)

        heith = image.shape[0]
        width = image.shape[1]

        if width < heith:
            h_new = 512
            w_new = int(width * h_new / heith)

        if width >= heith:
            w_new = 512
            h_new = int(heith * w_new / width)

        size = (h_new, w_new)

        image = tf.expand_dims(image, axis=0)
        resize_img = tf.image.resize_bicubic(image, size)

        scale_image = resize_img / 255
        scale_image = tf.clip_by_value(scale_image,
                                       clip_value_min=0,
                                       clip_value_max=1)

        return scale_image

    def load_model(self):
        """
        model used to calculate cost
        """

        # load model vgg without classifier
        vgg = tf.keras.applications.VGG19(include_top=False,)

        # replacing max pool by average pool
        vgg.save("base_model")
        custom_objects = {"MaxPooling2D": tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model("base_model",
                                         custom_objects=custom_objects)

        # save outputs of the VGG19 layers - style and content
        style_list = []
        content_list = []

        for layer in vgg.layers:
            layer.trainable = False
            if layer.name in self.content_layer:
                content_list.append(layer.output)
            if layer.name in self.style_layers:
                style_list.append(layer.output)

        model_outputs = style_list + content_list

        # Create model
        model = tf.keras.models.Model(vgg.input, model_outputs)

        self.model = model
