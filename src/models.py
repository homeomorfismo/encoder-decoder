"""
Implementation of basic linear models for encoder-decoder architectures
using JAX.
"""

import jax.numpy as jnp
from jax import jit


def __assert_sizes__(
    x: jnp.ndarray, weights: jnp.ndarray, bias: jnp.ndarray
) -> None:
    # TODO
    pass


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
    range_x = jnp.dot(fine_x, range_weights.T)
    return range_x
