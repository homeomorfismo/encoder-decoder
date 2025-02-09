"""
Solver module for iterative methods.
Implements Gauss-Seidel and symmetric Gauss-Seidel methods.
Using JAX for automatic differentiation and JIT compilation.
"""

import jax.numpy as jnp
from jax import jit, lax
from typing import Tuple

# Parameters
__TOL__: float = 1e-6
__MAX_ITER__: int = 1_000


@jit
def coordinate_descent(
    matrix: jnp.ndarray, x: jnp.ndarray, r: jnp.ndarray, i: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    t = r[i] / matrix[i, i]  # Compute the step size
    x = x.at[i].add(t)  # Update x[i]
    r = r - t * matrix[:, i]  # Update the residual
    return x, r


def substitution(
    matrix: jnp.ndarray, x: jnp.ndarray, b: jnp.ndarray, forward: bool
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r = b - jnp.dot(matrix, x)
    n = len(x)
    indices = jnp.arange(n) if forward else jnp.arange(n - 1, -1, -1)

    def body_fun(
        i: int, val: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x, r = val
        x, r = coordinate_descent(matrix, x, r, indices[i])
        return x, r

    x, r = lax.fori_loop(0, n, body_fun, (x, r))
    return x, r


def gauss_seidel_iteration(
    matrix: jnp.ndarray,
    x: jnp.ndarray,
    b: jnp.ndarray,
    tol: float,
    max_iter: int,
    forward: bool,
) -> jnp.ndarray:
    def body_fun(
        val: Tuple[int, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
        i, x, r = val
        x, r = substitution(matrix, x, b, forward)
        i += 1
        return i, x, r

    def cond_fun(val: Tuple[int, jnp.ndarray, jnp.ndarray]) -> bool:
        i, _, r = val
        return jnp.logical_and(jnp.linalg.norm(r) >= tol, i < max_iter)

    iter_count = 0
    x = jnp.zeros_like(b)
    r = b - jnp.dot(matrix, x)
    _, x, _ = lax.while_loop(cond_fun, body_fun, (iter_count, x, r))
    return x


@jit
def forward_gauss_seidel(
    matrix: jnp.ndarray,
    x: jnp.ndarray,
    b: jnp.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1_000,
) -> jnp.ndarray:
    return gauss_seidel_iteration(matrix, x, b, tol, max_iter, forward=True)


@jit
def backward_gauss_seidel(
    matrix: jnp.ndarray,
    x: jnp.ndarray,
    b: jnp.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1_000,
) -> jnp.ndarray:
    return gauss_seidel_iteration(matrix, x, b, tol, max_iter, forward=False)


@jit
def symmetric_gauss_seidel(
    matrix: jnp.ndarray,
    x: jnp.ndarray,
    b: jnp.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1_000,
) -> jnp.ndarray:
    def body_fun(
        val: Tuple[int, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
        i, x, r = val
        x, r = substitution(matrix, x, b, forward=True)
        x, r = substitution(matrix, x, b, forward=False)
        i += 1
        return i, x, r

    def cond_fun(val: Tuple[int, jnp.ndarray, jnp.ndarray]) -> bool:
        i, _, r = val
        return jnp.logical_and(jnp.linalg.norm(r) >= tol, i < max_iter)

    iter_count = 0
    x = jnp.zeros_like(b)
    r = b - jnp.dot(matrix, x)
    _, x, _ = lax.while_loop(cond_fun, body_fun, (iter_count, x, r))
    return x


@jit
def encoder_decoder_tl(
    fine_operator: jnp.ndarray,
    coarse_operator: jnp.ndarray,
    fine_to_coarse: jnp.ndarray,
    coarse_to_fine: jnp.ndarray,
    rhs: jnp.ndarray,
    solver_tol: float = 1e-6,
    solver_max_iter: int = 1_000,
    smoother_tol: float = 1e-1,
    smoother_max_iter: int = 5,
) -> jnp.ndarray:
    def body_fun(
        values: Tuple[int, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
        i, x_fine, x_coarse = values
        # Pre-smoothing: forward Gauss-Seidel
        residual_fine = rhs - jnp.dot(fine_operator, x_fine)
        x_fine = forward_gauss_seidel(
            fine_operator,
            x_fine,
            residual_fine,
            tol=smoother_tol,
            max_iter=smoother_max_iter,
        )
        # Coarse grid correction
        residual_coarse = jnp.dot(fine_to_coarse, residual_fine)
        x_coarse = jnp.linalg.solve(coarse_operator, residual_coarse)
        x_fine = x_fine + jnp.dot(coarse_to_fine, x_coarse)
        # Post-smoothing: backward Gauss-Seidel
        residual_fine = rhs - jnp.dot(fine_operator, x_fine)
        x_fine = backward_gauss_seidel(
            fine_operator,
            x_fine,
            rhs,
            tol=smoother_tol,
            max_iter=smoother_max_iter,
        )
        i += 1
        return i, x_fine, x_coarse

    def cond_fun(val: Tuple[int, jnp.ndarray, jnp.ndarray]) -> bool:
        i, x_fine, _ = val
        residual_fine = rhs - jnp.dot(fine_operator, x_fine)
        return jnp.logical_and(
            jnp.linalg.norm(residual_fine) >= solver_tol,
            i < solver_max_iter,
        )

    iter_count = 0
    x_fine = jnp.zeros_like(rhs)
    x_coarse = jnp.zeros(coarse_operator.shape[0], dtype=rhs.dtype)
    _, x_fine, _ = lax.while_loop(
        cond_fun, body_fun, (iter_count, x_fine, x_coarse)
    )
    return x_fine
