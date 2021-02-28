#!/usr/bin/env python3
"""YOLO object detection"""


import tensorflow.keras as K
import tensorflow as tf


class Yolo:
    """A YOLO object detection class"""
    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path) as class_file:
            self.class_names = [line.rstrip('\n') for line in class_file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
