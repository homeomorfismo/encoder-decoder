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


def l1_smoother():
    """
    Solve the linear system Ax = b using the l1 smoother method.
    """
    pass


# Test functions
def test_forward_gauss_seidel():
    """
    Test the Gauss-Seidel method with forward substitution on a dense matrix.
    """
    # Matrix needs to be SPD for Gauss-Seidel to converge
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.dot(matrix, 2.0 * np.ones(2))
    x = np.random.rand(2)
    x = forward_gauss_seidel(matrix, x, b, tol=1e-6, max_iter=1000, verbose=True)
    assert np.allclose(x, 2.0 * np.ones(2))


def test_backward_gauss_seidel():
    """
    Test the Gauss-Seidel method with backward substitution on a dense matrix.
    """
    # Matrix needs to be SPD for Gauss-Seidel to converge
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.dot(matrix, 2.0 * np.ones(2))
    x = np.random.rand(2)
    x = backward_gauss_seidel(matrix, x, b, tol=1e-6, max_iter=1000, verbose=True)
    assert np.allclose(x, 2.0 * np.ones(2))


def test_symmetric_gauss_seidel():
    """
    Test the Gauss-Seidel method with forward and backward substitution on a dense matrix.
    """
    # Matrix needs to be SPD for Gauss-Seidel to converge
    matrix = 4.0 * np.eye(20) + np.diag(np.ones(19), 1) + np.diag(np.ones(19), -1)
    b = np.dot(matrix, 2.0 * np.ones(20))
    x = np.random.rand(20)
    x = symmetric_gauss_seidel(matrix, x, b, tol=1e-6, max_iter=1000, verbose=True)
    assert np.allclose(x, 2.0 * np.ones(20))


if __name__ == "__main__":
    print("Forward Gauss-Seidel")
    test_forward_gauss_seidel()
    print("Backward Gauss-Seidel")
    test_backward_gauss_seidel()
    print("Symmetric Gauss-Seidel")
    test_symmetric_gauss_seidel()
    print("All tests passed.")
