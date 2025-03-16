"""
Implementation of (batched) CSR-matrix-vector multiplication using JAX.
Subsequent implementation of basic linear models for encoder-decoder
architectures using JAX.

The spirit of this module is to auto-diff the models in the data variable of
the CSR matrix, while fixing the sparsity of the matrix (i.e. the indices of
the initially non-zero elements).

IMPORTANT:
This module might be subject to change in the future, due to current work on
jax.experimental.sparse, which could provide a more efficient way to handle
sparse matrices in JAX and in this work.
"""

from typing import Tuple, Optional
from functools import partial
from jax import jit
import jax.numpy as jnp
import jax.lax as lax


def __sparse_matvec(
    x: jnp.ndarray,
    weights_data: jnp.ndarray,
    weights_col_idx: jnp.ndarray,
    weights_row_ptr: jnp.ndarray,
    output_dim: int,
) -> jnp.ndarray:
    """
    Sparse matrix-vector multiplication for CSR format matrices.
    Compatible with JIT compilation.

    Args:
        x: Input vector of shape (..., n)
        weights_data: Non-zero values in the CSR matrix
        weights_col_idx: Column indices for non-zero values
        weights_row_ptr: Row pointers indicating start/end positions
        output_dim: Output dimension for the result

    Returns:
        Result of matrix-vector multiplication with shape (..., output_dim)
    """
    # Create output with provided shape
    y_shape = (*x.shape[:-1], output_dim)
    y = jnp.zeros(y_shape, dtype=x.dtype)

    # Define a function to process a single row
    def process_row(row_idx, acc_y):
        # Get row bounds
        row_start = weights_row_ptr[row_idx]
        row_end = weights_row_ptr[row_idx + 1]
        x_i = x[..., row_idx]

        # Process non-zero elements in this row
        def process_element(j, y_acc):
            col = weights_col_idx[j]
            val = weights_data[j]
            # Update output at column position
            y_acc = y_acc.at[..., col].add(x_i * val)
            return y_acc

        # Process all elements in the row
        updated_y = lax.fori_loop(row_start, row_end, process_element, acc_y)
        return updated_y

    # Process all rows
    n_rows = len(weights_row_ptr) - 1
    y = lax.fori_loop(0, n_rows, process_row, y)
    return y


@partial(jit, static_argnums=(5,))
def SparseLinearLayer(
    x: jnp.ndarray,
    weights_data: jnp.ndarray,
    weights_col_idx: jnp.ndarray,
    weights_row_ptr: jnp.ndarray,
    bias: jnp.ndarray,
    output_dim: Optional[int] = None,
) -> jnp.ndarray:
    """
    Sparse linear layer x -> Wx + b.
    JIT-compiled with output_dim as a static argument.

    Args:
        x: Input tensor of shape (..., input_dim)
        weights_data: Non-zero values in the CSR matrix
        weights_col_idx: Column indices for non-zero values
        weights_row_ptr: Row pointers indicating start/end positions
        bias: Bias vector of shape (output_dim,)
        output_dim: Output dimension (required for JIT compilation)

    Returns:
        Output tensor of shape (..., output_dim)
    """
    # If output_dim is not provided, we can't use the JIT-compiled path
    if output_dim is None:
        # This path won't be JIT-compiled
        output_dim = bias.shape[0]

    # Apply sparse matrix multiplication
    y = __sparse_matvec(
        x, weights_data, weights_col_idx, weights_row_ptr, output_dim
    )

    # Add bias
    return y + bias
