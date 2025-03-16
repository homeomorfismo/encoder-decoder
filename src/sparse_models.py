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

from typing import Optional
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
    y_shape = (*x.shape[:-1], output_dim)
    y = jnp.zeros(y_shape, dtype=x.dtype)

    def _process_row(row_idx, acc_y):
        row_start = weights_row_ptr[row_idx]
        row_end = weights_row_ptr[row_idx + 1]
        x_i = x[..., row_idx]

        def _process_element(j, y_acc):
            col = weights_col_idx[j]
            val = weights_data[j]
            y_acc = y_acc.at[..., col].add(x_i * val)
            return y_acc

        updated_y = lax.fori_loop(row_start, row_end, _process_element, acc_y)
        return updated_y

    n_rows = len(weights_row_ptr) - 1
    y = lax.fori_loop(0, n_rows, _process_row, y)
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
        output_dim = bias.shape[0]
    y = __sparse_matvec(
        x,
        weights_data,
        weights_col_idx,
        weights_row_ptr,
        output_dim,
    )

    return y + bias


@partial(jit, static_argnums=(6, 7))
def SparseLinearEncoderDecoder(
    x: jnp.ndarray,
    encoder_data: jnp.ndarray,
    encoder_col_idx: jnp.ndarray,
    encoder_row_ptr: jnp.ndarray,
    decoder_data: jnp.ndarray,
    decoder_col_idx: jnp.ndarray,
    decoder_row_ptr: jnp.ndarray,
    output_encoder_dim: Optional[int] = None,
    output_decoder_dim: Optional[int] = None,
) -> jnp.ndarray:
    """
    Encoder-Decoder architecture using sparse linear layers.
    Null biases are used.
    JIT-compiled with output dimensions as static arguments.
    """
    # If output dimensions are not provided, we can't use the JIT-compiled path
    if output_encoder_dim is None:
        output_encoder_dim = encoder_row_ptr.shape[0] - 1
    if output_decoder_dim is None:
        output_decoder_dim = decoder_row_ptr.shape[0] - 1

    coarse_x = SparseLinearLayer(
        x,
        encoder_data,
        encoder_col_idx,
        encoder_row_ptr,
        jnp.zeros(output_encoder_dim, dtype=x.dtype),
        output_dim=output_encoder_dim,
    )
    fine_x = SparseLinearLayer(
        coarse_x,
        decoder_data,
        decoder_col_idx,
        decoder_row_ptr,
        jnp.zeros(output_decoder_dim, dtype=x.dtype),
        output_dim=output_decoder_dim,
    )

    return fine_x


@partial(jit, static_argnums=(7, 8, 9))
def SparseMGLinearEncoderDecoder(
    x: jnp.ndarray,
    encoder_data: jnp.ndarray,
    encoder_col_idx: jnp.ndarray,
    encoder_row_ptr: jnp.ndarray,
    decoder_data: jnp.ndarray,
    decoder_col_idx: jnp.ndarray,
    decoder_row_ptr: jnp.ndarray,
    range_data: jnp.ndarray,
    range_col_idx: jnp.ndarray,
    range_row_ptr: jnp.ndarray,
    output_encoder_dim: Optional[int] = None,
    output_decoder_dim: Optional[int] = None,
    output_range_dim: Optional[int] = None,
) -> jnp.ndarray:
    """
    MG Encoder-Decoder architecture using sparse linear layers.
    Null biases are used.
    JIT-compiled with output dimensions as static arguments.
    """
    # If output dimensions are not provided, we can't use the JIT-compiled path
    if output_encoder_dim is None:
        output_encoder_dim = encoder_row_ptr.shape[0] - 1
    if output_decoder_dim is None:
        output_decoder_dim = decoder_row_ptr.shape[0] - 1
    if output_range_dim is None:
        output_range_dim = range_row_ptr.shape[0] - 1

    coarse_x = SparseLinearLayer(
        x,
        encoder_data,
        encoder_col_idx,
        encoder_row_ptr,
        jnp.zeros(output_encoder_dim, dtype=x.dtype),
        output_dim=output_encoder_dim,
    )
    fine_x = SparseLinearLayer(
        coarse_x,
        decoder_data,
        decoder_col_idx,
        decoder_row_ptr,
        jnp.zeros(output_decoder_dim, dtype=x.dtype),
        output_dim=output_decoder_dim,
    )
    range_x = SparseLinearLayer(
        fine_x,
        range_data,
        range_col_idx,
        range_row_ptr,
        jnp.zeros(output_range_dim, dtype=x.dtype),
        output_dim=output_range_dim,
    )
    return range_x
