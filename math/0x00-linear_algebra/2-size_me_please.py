#!/usr/bin/env python3

def matrix_shape(matrix):
    shape=[]
    mat=matrix
    i = len(mat)
    while i > 0:
        shape.append(i)
        mat=mat[0]
        if isinstance(mat, list):
            i=len(mat)
        else:
            i = 0
    return shape
