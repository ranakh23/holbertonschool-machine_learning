#!/usr/bin/env python3
""" DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """ class DeepNeuralNetwork """
    def __init__(self, nx, layers):
        """ init for DeepNeuralNetwork """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list:
            raise TypeError('layers must be a list of positive integers')
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for ly in range(self.L):
            if type(layers[ly]) != int or layers[ly] <= 0:
                raise TypeError('layers must be a list of positive integers')
            self.weights["b"+str(ly+1)] = np.zeros((layers[ly], 1))
            if ly == 0:
                heetal = np.random.randn(layers[ly], nx) * np.sqrt(2/nx)
                self.weights["W"+str(ly+1)] = heetal
            else:
                factor = np.sqrt(2/layers[ly-1])
                heetal = np.random.randn(layers[ly], layers[ly-1]) * factor
                self.weights["W" + str(ly+1)] = heetal
