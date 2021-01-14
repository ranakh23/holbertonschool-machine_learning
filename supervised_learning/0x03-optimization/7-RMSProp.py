#!/usr/bin/env python3
""" RMSProp """


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ updates a variable using the RMSProp optimization algorithm """
    Sdv = beta2*s + (1-beta2)*(grad**2)
    var_updated = var - alpha*(grad/(Sdv**(1/2)+epsilon))
    return var_updated, Sdv
