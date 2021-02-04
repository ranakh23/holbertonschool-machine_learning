#!/usr/bin/env python3
"""function lenet5. """
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras.
    """
    He = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                            activation='relu', kernel_initializer=He)(X)
    pool1 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                            activation='relu', kernel_initializer=He)(pool1)
    pool2 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    pool2 = K.layers.Flatten()(pool2)
    dense1 = K.layers.Dense(units=120, activation='relu',
                            kernel_initializer=He)(pool2)
    dense2 = K.layers.Dense(units=84, activation='relu',
                            kernel_initializer=He)(dense1)
    dense3 = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=He)(dense2)
    network = K.Model(inputs=X, outputs=dense3)
    network.compile(optimizer=K.optimizers.Adam(),
                    loss='categorical_crossentropy', metrics=['accuracy'])
    return network
