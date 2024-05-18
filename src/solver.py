"""
Linear algebra solvers
"""

import numpy as np


# Dense matrix solvers
def coordinate_descent(matrix, x, r, i):
    """
    Do coordinate descent on the linear system Ax = b,
    for a given guess x and residual r = b - Ax.
    Assumes that matrix is dense.
    """
    t = r[i] / matrix[i, i]  # Compute the step size
    x[i] += t  # Update x[i]
    temp = np.ravel(t * matrix[:, i])  # Compute the update to the residual
    r -= temp  # Update the residual
    return x, r


def forward_substitution(matrix, x, b):
    """
    Do forward substitution on the linear system Ax = b,
    for a given guess x and residual r = b - Ax.
    Assumes that matrix is dense.
    """
    r = np.ravel(b - matrix @ x)
    for i in range(len(x)):
        x, r = coordinate_descent(matrix, x, r, i)
    return x, r


def backward_substitution(matrix, x, b):
    """
    Do backward substitution on the linear system Ax = b,
    for a given guess x and residual r = b - Ax.
    Assumes that matrix is dense.
    """
    r = np.ravel(b - matrix @ x)
    for i in range(len(x) - 1, -1, -1):
        x, r = coordinate_descent(matrix, x, r, i)
    return x, r


def forward_gauss_seidel(matrix, x, b, tol=1e-6, max_iter=1000, verbose=False):
    """
    Solve the linear system Ax = b using the Gauss-Seidel method.
    Assumes that matrix is dense.
    """
    for i in range(max_iter):
        x, r = forward_substitution(matrix, x, b)
        if np.linalg.norm(r) < tol:
            break
        if verbose:
            print(f"Iteration {i}: norm = {np.linalg.norm(r)}")
    return x


def backward_gauss_seidel(matrix, x, b, tol=1e-6, max_iter=1000, verbose=False):
    """
    Solve the linear system Ax = b using the Gauss-Seidel method.
    Assumes that matrix is dense.
    """
    for i in range(max_iter):
        x, r = backward_substitution(matrix, x, b)
        if np.linalg.norm(r) < tol:
            break
        if verbose:
            print(f"Iteration {i}: norm = {np.linalg.norm(r)}")
    return x


def symmetric_gauss_seidel(matrix, x, b, tol=1e-6, max_iter=1000, verbose=False):
    """
    Solve the linear system Ax = b using the Gauss-Seidel method.
    """
    for i in range(max_iter):
        x, r = forward_substitution(matrix, x, b)
        x, r = backward_substitution(matrix, x, b)
        if np.linalg.norm(r) < tol:
            break
        if verbose:
            print(f"Iteration {i}: norm = {np.linalg.norm(r)}")
    return x


# Test functions
def test_forward_gauss_seidel():
    """
    Test the Gauss-Seidel method with forward substitution on a dense matrix.
    """
    # Matrix needs to be SPD for Gauss-Seidel to converge
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.dot(matrix, np.array([1.0, 1.0]))
    x = np.zeros(2)
    x = forward_gauss_seidel(matrix, x, b, tol=1e-6, max_iter=1000, verbose=True)
    assert np.allclose(x, np.array([1.0, 1.0]))


def test_backward_gauss_seidel():
    """
    Test the Gauss-Seidel method with backward substitution on a dense matrix.
    """
    # Matrix needs to be SPD for Gauss-Seidel to converge
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.dot(matrix, np.array([1.0, 1.0]))
    x = np.zeros(2)
    x = backward_gauss_seidel(matrix, x, b, tol=1e-6, max_iter=1000, verbose=True)
    assert np.allclose(x, np.array([1.0, 1.0]))


if __name__ == "__main__":
    test_forward_gauss_seidel()
    test_backward_gauss_seidel()
    print("All tests passed.")
