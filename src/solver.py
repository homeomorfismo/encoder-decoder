"""
Linear algebra solvers
"""

import numpy as np


def forward_substitution(L, b):
    """
    Forward substitution solver for a lower triangular matrix L
    """
    n = len(b)
    y = np.zeros(n)
    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def backward_substitution(U, b):
    """
    Backward substitution solver for an upper triangular matrix U
    """
    n = len(b)
    x = np.zeros(n)
    x[-1] = b[-1] / U[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x
