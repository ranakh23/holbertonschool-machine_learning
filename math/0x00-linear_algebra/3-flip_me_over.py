#!/usr/bin/env python3
"get transpose"


def matrix_transpose(matrix):
    "fun fun function"
    mat = []
    for j in range(len(matrix[0])):
        col = []
        for i in matrix:
            col.append(i[j])
        mat.append(col)
    return mat
