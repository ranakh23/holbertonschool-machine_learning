#!/usr/bin/env python3
""" Moving Average """


def moving_average(data, beta):
    """ calculates the weighted moving average of a data set """
    avg = []
    vt = 0
    for i in range(len(data)):
        vt = beta*vt + (1-beta)*data[i]
        avg.append(vt/(1-beta**(i+1)))
    return avg
