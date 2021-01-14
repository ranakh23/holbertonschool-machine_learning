#!/usr/bin/env python3
""" Adam """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ updates a variable in place using the Adam optimization algorithm """
    Vdv = beta1*v + (1-beta1)*grad
    Sdv = beta2*s + (1-beta2)*(grad)**2

    Vdv_h = Vdv / (1-(beta1**t))
    Sdv_h = Sdv / (1-(beta2**t))
    var = var - alpha*(Vdv_h/((Sdv_h**(1/2))+epsilon))
    return var, Vdv, Sdv
