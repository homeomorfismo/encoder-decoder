"""
Implementation of basic linear models for encoder-decoder architectures
using JAX.
"""

import jax
import jax.numpy as jnp
from jax import random, jit

# Parameters
__SEED__: int = 0
__NUM_EPOCHS__: int = 100
__BATCH_SIZE__: int = 100
__LEARNING_RATE__: float = 0.01
__INIT_NAME__: str = "glorot_uniform"


def __assert_sizes__(
    x: jnp.ndarray, weights: jnp.ndarray, bias: jnp.ndarray
) -> None:
    # assert x.ndim == 2, "Input must be 2D"
    assert weights.ndim == 2, "Weights must be 2D"
    assert bias.ndim == 1, "Bias must be 1D"
    assert (
        x.shape[-1] == weights.shape[0]
    ), "Input and weights dimensions mismatch"
    assert (
        weights.shape[1] == bias.shape[0]
    ), "Weights and bias dimensions mismatch"


@jit
def LinearLayer(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    """
    Simple linear layer x -> Wx + b.
    """
    __assert_sizes__(x, weights, bias)
    return jnp.dot(x, weights) + bias


@jit
def LinearEncoderDecoder(
    x: jnp.ndarray,
    encoder_weights: jnp.ndarray,
    decoder_weights: jnp.ndarray,
) -> jnp.ndarray:
    """
    Encoder-Decoder architecture using linear layers.
    Null biases are used.
    """
    coarse_x = LinearLayer(
        x,
        encoder_weights,
        jnp.zeros(encoder_weights.shape[1], dtype=x.dtype),
    )
    fine_x = LinearLayer(
        coarse_x,
        decoder_weights,
        jnp.zeros(decoder_weights.shape[1], dtype=x.dtype),
    )
    return fine_x


@jit
def MGLinearEncoderDecoder(
    x: jnp.ndarray,
    encoder_weights: jnp.ndarray,
    decoder_weights: jnp.ndarray,
    range_weights: jnp.ndarray,
) -> jnp.ndarray:
    """
    MG Encoder-Decoder architecture using linear layers.
    Null biases are used.
    """
    coarse_x = LinearLayer(
        x,
        encoder_weights,
        jnp.zeros(encoder_weights.shape[1], dtype=x.dtype),
    )
    fine_x = LinearLayer(
        coarse_x,
        decoder_weights,
        jnp.zeros(decoder_weights.shape[1], dtype=x.dtype),
    )
    range_x = LinearLayer(
        fine_x,
        range_weights,
        jnp.zeros(range_weights.shape[1], dtype=x.dtype),
    )
    return range_x
