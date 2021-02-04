#!/usr/bin/env python3
"""Train a keras model"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """Train a keras model"""
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))
    if learning_rate_decay and validation_data:
        def __schedule(epoch):
            """Scale the learning rate based on epoch"""
            return alpha * 1 / (epoch * decay_rate + 1)
        callbacks.append(K.callbacks.LearningRateScheduler(__schedule, 1))
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       shuffle=shuffle, verbose=verbose,
                       validation_data=validation_data,
                       callbacks=callbacks)
