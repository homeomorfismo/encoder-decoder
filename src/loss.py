"""
Implementation of loss functions for encoder-decoder architectures
using JAX.
"""

import jax.numpy as jnp
from jax import jit
from typing import Callable, Union

# local imports
from models import LinearEncoderDecoder, MGLinearEncoderDecoder


def get_loss(ord: Union[int, float, str, None]) -> Callable:
    """
    Get the loss function based on the specified type.

    Args:
    - ord (Union[int, float, jnp.inf, -jnp.inf, 'fro', 'nuc', None]): The type of loss function to use.

    Returns:
    - Callable: The JAX loss function.
    """

    @jit
    def loss(
        x: jnp.ndarray,
        encoder_weights: jnp.ndarray,
        decoder_weights: jnp.ndarray,
        reg: float,
    ) -> float:
        """
        Loss function for encoder-decoder architecture.
        * MSE loss for reconstruction.
        * Regularization on encoder and decoder weights.
        """
        reconstr_loss = jnp.mean(
            (x - LinearEncoderDecoder(x, encoder_weights, decoder_weights))
            ** 2
        )
        if ord == 0:
            reg_loss = jnp.sum(jnp.abs(encoder_weights) > 0)
            reg_loss += jnp.sum(jnp.abs(decoder_weights) > 0)
            reg_loss += jnp.sum(
                jnp.abs(
                    jnp.eye(decoder_weights.shape[0])
                    - jnp.dot(decoder_weights, encoder_weights)
                )
                > 0
            )
        else:
            reg_loss = jnp.linalg.norm(encoder_weights, ord=ord)
            reg_loss += jnp.linalg.norm(decoder_weights, ord=ord)
            reg_loss += jnp.linalg.norm(
                jnp.eye(decoder_weights.shape[0])
                - jnp.dot(decoder_weights, encoder_weights),
                ord=ord,
            )
        return reconstr_loss + reg * reg_loss

    return loss


def get_mg_loss(ord: Union[int, float, str, None]) -> Callable:
    """
    Get the loss function based on the specified type.

    Args:
    - ord (Union[int, float, inf, -inf, 'fro', 'nuc', None]): The type of loss function to use.

    Returns:
    - Callable: The JAX loss function.
    """

    @jit
    def loss_mg(
        x: jnp.ndarray,
        encoder_weights: jnp.ndarray,
        decoder_weights: jnp.ndarray,
        range_weights: jnp.ndarray,
        reg: float,
    ) -> float:
        """
        Loss function for MG encoder-decoder architecture.
        * MSE loss for reconstruction.
        * Regularization on encoder and decoder weights.
        """
        reconstr_loss = jnp.mean(
            jnp.dot(
                x
                - MGLinearEncoderDecoder(
                    x, encoder_weights, decoder_weights, range_weights
                ),
                jnp.transpose(decoder_weights),
            )
            ** 2
        )
        if ord == 0:
            reg_loss = jnp.sum(jnp.abs(encoder_weights) > 0)
            reg_loss += jnp.sum(jnp.abs(decoder_weights) > 0)
            reg_loss += jnp.sum(
                jnp.abs(
                    jnp.eye(decoder_weights.shape[0])
                    - jnp.dot(decoder_weights, encoder_weights)
                )
                > 0
            )
        else:
            reg_loss = jnp.linalg.norm(encoder_weights, ord=ord)
            reg_loss += jnp.linalg.norm(decoder_weights, ord=ord)
            reg_loss += jnp.linalg.norm(
                jnp.eye(decoder_weights.shape[0])
                - jnp.dot(decoder_weights, encoder_weights),
                ord=ord,
            )
        return reconstr_loss + reg * reg_loss

    return loss_mg


def __test_loss_functions():
    import numpy as np

    x = np.random.rand(10, 10)
    encoder_weights = np.random.rand(10, 10)
    decoder_weights = np.random.rand(10, 10)
    range_weights = np.random.rand(10, 10)
    reg = 0.1

    ord_list = [0, 1, 2, jnp.inf, -jnp.inf, "fro", "nuc", None]

    for ord in ord_list:
        loss = get_loss(ord)
        loss_mg = get_mg_loss(ord)
        print(f"Testing loss function with ord={ord}")
        print(f"Loss: {loss(x, encoder_weights, decoder_weights, reg)}")
        print(
            f"MG Loss: {loss_mg(x, encoder_weights, decoder_weights, range_weights, reg)}"
        )


if __name__ == "__main__":
    __test_loss_functions()
