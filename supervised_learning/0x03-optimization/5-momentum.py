#!/usr/bin/env python3
""" Momentum """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ updates a variable using the GD with momentum opt algorithm """
    Vdv = beta1*v + (1-beta1)*grad
    var_updated = var - alpha*Vdv
    return var_updated, Vdv
