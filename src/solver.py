"""
Linear algebra solvers using JAX
"""

import jax.numpy as jnp
from jax import jit
import scipy as sp
from typing import Tuple


# Dense matrix solvers
@jit
def coordinate_descent(
    matrix: jnp.ndarray, x: jnp.ndarray, r: jnp.ndarray, i: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Do coordinate descent on the linear system Ax = b,
    for a given guess x and residual r = b - Ax.
    Assumes that matrix is dense.
    """
    t = r[i] / matrix[i, i]  # Compute the step size
    x = x.at[i].add(t)  # Update x[i]
    temp = t * matrix[:, i]  # Compute the update to the residual
    r = r - temp  # Update the residual
    return x, r


def forward_substitution(
    matrix: jnp.ndarray, x: jnp.ndarray, b: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Do forward substitution on the linear system Ax = b,
    for a given guess x and residual r = b - Ax.
    Assumes that matrix is dense.
    """
    r = b - jnp.dot(matrix, x)
    for i in range(len(x)):
        x, r = coordinate_descent(matrix, x, r, i)
    return x, r


def backward_substitution(
    matrix: jnp.ndarray, x: jnp.ndarray, b: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Do backward substitution on the linear system Ax = b,
    for a given guess x and residual r = b - Ax.
    Assumes that matrix is dense.
    """
    r = b - jnp.dot(matrix, x)
    for i in range(len(x) - 1, -1, -1):
        x, r = coordinate_descent(matrix, x, r, i)
    return x, r


def forward_gauss_seidel(
    matrix: jnp.ndarray,
    x: jnp.ndarray,
    b: jnp.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1_000,
    verbose: bool = False,
) -> jnp.ndarray:
    """
    Solve the linear system Ax = b using the Gauss-Seidel method.
    Assumes that matrix is dense.
    """
    for i in range(int(max_iter)):
        x, r = forward_substitution(matrix, x, b)
        if jnp.linalg.norm(r) < tol:
            break
        if verbose:
            print(f"Iteration {i}: norm = {jnp.linalg.norm(r)}")
    return x


def backward_gauss_seidel(
    matrix: jnp.ndarray,
    x: jnp.ndarray,
    b: jnp.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1_000,
    verbose: bool = False,
) -> jnp.ndarray:
    """
    Solve the linear system Ax = b using the Gauss-Seidel method.
    Assumes that matrix is dense.
    """
    for i in range(int(max_iter)):
        x, r = backward_substitution(matrix, x, b)
        if jnp.linalg.norm(r) < tol:
            break
        if verbose:
            print(f"Iteration {i}: norm = {jnp.linalg.norm(r)}")
    return x


def symmetric_gauss_seidel(
    matrix: jnp.ndarray,
    x: jnp.ndarray,
    b: jnp.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1_000,
    verbose: bool = False,
) -> jnp.ndarray:
    """
    Solve the linear system Ax = b using the Gauss-Seidel method.
    """
    for i in range(int(max_iter)):
        x, r = forward_substitution(matrix, x, b)
        x, r = backward_substitution(matrix, x, b)
        if jnp.linalg.norm(r) < tol:
            break
        if verbose:
            print(f"Iteration {i}: norm = {jnp.linalg.norm(r)}")
    return x


# TODO: Two-level wrapper
@jit
def encoder_decoder_tl(
    fine_operator: jnp.ndarray,
    coarse_operator: jnp.ndarray,
    fine_to_coarse: jnp.ndarray,
    coarse_to_fine: jnp.ndarray,
    rhs: jnp.ndarray,
    solver_tol: float = 1e-6,
    solver_max_iter: int = 1_000,
    verbose: bool = False,
) -> jnp.ndarray:
    """
    Two-level solver for a linear system Ax = b.
    We assume b is trivial on boundary dofs.
    """
    x_fine = jnp.zeros_like(rhs)
    x_coarse = jnp.zeros_like(coarse_operator @ rhs)
    for i in range(int(solver_max_iter)):
        # Pre-smoothing: forward Gauss-Seidel
        residual_fine = rhs - fine_operator @ x_fine
        x_fine = forward_gauss_seidel(
            fine_operator,
            x_fine,
            rhs,
            tol=solver_tol,
            max_iter=1,
            verbose=False,
        )
        # Coarse grid correction
        residual_coarse = fine_to_coarse @ residual_fine
        x_coarse = sp.linalg.solve(coarse_operator, residual_coarse)
        x_fine = x_fine + coarse_to_fine @ x_coarse
        # Post-smoothing: backward Gauss-Seidel
        residual_fine = rhs - fine_operator @ x_fine
        x_fine = backward_gauss_seidel(
            fine_operator,
            x_fine,
            rhs,
            tol=solver_tol,
            max_iter=1,
            verbose=False,
        )
        if jnp.linalg.norm(residual_fine) < solver_tol:
            break
        if verbose:
            print(f"Iteration {i}: norm = {jnp.linalg.norm(residual_fine)}")
    return x_fine


# Test functions
def test_forward_gauss_seidel() -> None:
    """
    Test the Gauss-Seidel method with forward substitution on a dense matrix.
    """
    matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    b = jnp.dot(matrix, 2.0 * jnp.ones(2))
    x = jnp.array([1.0, 1.0])  # Initial guess
    # Ensure max_iter is a concrete value
    max_iter = 1000
    x = forward_gauss_seidel(
        matrix, x, b, tol=1e-6, max_iter=max_iter, verbose=True
    )
    assert jnp.allclose(x, 2.0 * jnp.ones(2))


def test_backward_gauss_seidel() -> None:
    """
    Test the Gauss-Seidel method with backward substitution on a dense matrix.
    """
    matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    b = jnp.dot(matrix, 2.0 * jnp.ones(2))
    x = jnp.array([1.0, 1.0])  # Initial guess
    # Ensure max_iter is a concrete value
    max_iter = 1_000
    x = backward_gauss_seidel(
        matrix, x, b, tol=1e-6, max_iter=max_iter, verbose=True
    )
    assert jnp.allclose(x, 2.0 * jnp.ones(2))


def test_symmetric_gauss_seidel() -> None:
    """
    Test the Gauss-Seidel method with forward and backward substitution on a dense matrix.
    """
    matrix = (
        4.0 * jnp.eye(20)
        + jnp.diag(jnp.ones(19), 1)
        + jnp.diag(jnp.ones(19), -1)
    )
    b = jnp.dot(matrix, 2.0 * jnp.ones(20))
    x = jnp.array([1.0] * 20)  # Initial guess
    # Ensure max_iter is a concrete value
    max_iter = 1_000
    x = symmetric_gauss_seidel(
        matrix, x, b, tol=1e-6, max_iter=max_iter, verbose=True
    )
    assert jnp.allclose(x, 2.0 * jnp.ones(20))


if __name__ == "__main__":
    print("Forward Gauss-Seidel")
    test_forward_gauss_seidel()
    print("Backward Gauss-Seidel")
    test_backward_gauss_seidel()
    print("Symmetric Gauss-Seidel")
    test_symmetric_gauss_seidel()
    print("All tests passed.")
