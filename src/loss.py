"""
Implementation of loss functions for encoder-decoder architectures
using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit


######################
# L2 Loss functions
######################
@jit
def loss_eucledian(
    x: jnp.ndarray,
    y: jnp.ndarray,
    encoder_weights: jnp.ndarray,
    decoder_weights: jnp.ndarray,
    reg: float,
) -> float:
    """
    Loss function for encoder-decoder architecture.
    * MSE loss for reconstruction.
    * L2 regularization on encoder and decoder weights.
    """
    reconstr_loss = jnp.mean((x - y) ** 2)
    reg_loss = jnp.linalg.norm(encoder_weights, ord=2)
    reg_loss += jnp.linalg.norm(decoder_weights, ord=2)
    reg_loss += jnp.linalg.norm(
        jnp.eye(decoder_weights.shape[0])
        - jnp.dot(decoder_weights, encoder_weights),
        ord=2,
    )
    return reconstr_loss + reg * reg_loss


@jit
def loss_mg_eucledian(
    x: jnp.ndarray,
    y: jnp.ndarray,
    encoder_weights: jnp.ndarray,
    decoder_weights: jnp.ndarray,
    reg: float,
) -> float:
    """
    Loss function for MG encoder-decoder architecture.
    * MSE loss for reconstruction.
    * L2 regularization on encoder and decoder weights.
    """
    reconstr_loss = jnp.mean(
        jnp.dot(x - y, jnp.transpose(decoder_weights)) ** 2
    )
    reg_loss = jnp.linalg.norm(encoder_weights, ord=2)
    reg_loss += jnp.linalg.norm(decoder_weights, ord=2)
    reg_loss += jnp.linalg.norm(
        jnp.eye(decoder_weights.shape[0])
        - jnp.dot(decoder_weights, encoder_weights),
        ord=2,
    )
    return reconstr_loss + reg * reg_loss


######################
# L1 Loss functions
######################
@jit
def loss_l1(
    x: jnp.ndarray,
    y: jnp.ndarray,
    encoder_weights: jnp.ndarray,
    decoder_weights: jnp.ndarray,
    reg: float,
) -> float:
    """
    Loss function for encoder-decoder architecture.
    * MSE loss for reconstruction.
    * L1 regularization on encoder and decoder weights.
    """
    reconstr_loss = jnp.mean((x - y) ** 2)
    reg_loss = jnp.linalg.norm(encoder_weights, ord=1)
    reg_loss += jnp.linalg.norm(decoder_weights, ord=1)
    reg_loss += jnp.linalg.norm(
        jnp.eye(decoder_weights.shape[0])
        - jnp.dot(decoder_weights, encoder_weights),
        ord=1,
    )
    return reconstr_loss + reg * reg_loss


@jit
def loss_mg_l1(
    x: jnp.ndarray,
    y: jnp.ndarray,
    encoder_weights: jnp.ndarray,
    decoder_weights: jnp.ndarray,
    reg: float,
) -> float:
    """
    Loss function for MG encoder-decoder architecture.
    * MSE loss for reconstruction.
    * L1 regularization on encoder and decoder weights.
    """
    reconstr_loss = jnp.mean(
        jnp.dot(x - y, jnp.transpose(decoder_weights)) ** 2
    )
    reg_loss = jnp.linalg.norm(encoder_weights, ord=1)
    reg_loss += jnp.linalg.norm(decoder_weights, ord=1)
    reg_loss += jnp.linalg.norm(
        jnp.eye(decoder_weights.shape[0])
        - jnp.dot(decoder_weights, encoder_weights),
        ord=1,
    )
    return reconstr_loss + reg * reg_loss


######################
# L0 Loss functions
######################
@jit
def loss_l0(
    x: jnp.ndarray,
    y: jnp.ndarray,
    encoder_weights: jnp.ndarray,
    decoder_weights: jnp.ndarray,
    reg: float,
) -> float:
    """
    Loss function for encoder-decoder architecture.
    * MSE loss for reconstruction.
    * L0 regularization on encoder and decoder weights.
    """
    reconstr_loss = jnp.mean((x - y) ** 2)
    reg_loss = jnp.sum(jnp.abs(encoder_weights) > 0)
    reg_loss += jnp.sum(jnp.abs(decoder_weights) > 0)
    reg_loss += jnp.sum(
        jnp.abs(
            jnp.eye(decoder_weights.shape[0])
            - jnp.dot(decoder_weights, encoder_weights)
        )
        > 0
    )
    return reconstr_loss + reg * reg_loss


@jit
def loss_mg_l0(
    x: jnp.ndarray,
    y: jnp.ndarray,
    encoder_weights: jnp.ndarray,
    decoder_weights: jnp.ndarray,
    reg: float,
) -> float:
    """
    Loss function for MG encoder-decoder architecture.
    * MSE loss for reconstruction.
    * L0 regularization on encoder and decoder weights.
    """
    reconstr_loss = jnp.mean(
        jnp.dot(x - y, jnp.transpose(decoder_weights)) ** 2
    )
    reg_loss = jnp.sum(jnp.abs(encoder_weights) > 0)
    reg_loss += jnp.sum(jnp.abs(decoder_weights) > 0)
    reg_loss += jnp.sum(
        jnp.abs(
            jnp.eye(decoder_weights.shape[0])
            - jnp.dot(decoder_weights, encoder_weights)
        )
        > 0
    )
    return reconstr_loss + reg * reg_loss


if __name__ == "__main__":
    # Test loss functions
    import numpy as np

    x = np.random.rand(10, 10)
    y = np.random.rand(10, 10)
    encoder_weights = np.random.rand(10, 10)
    decoder_weights = np.random.rand(10, 10)
    reg = 0.1

    names = [
        "Eucledian",
        "MG Eucledian",
        "L1",
        "MG L1",
        "L0",
        "MG L0",
    ]
    values = []
    values.append(loss_eucledian(x, y, encoder_weights, decoder_weights, reg))
    values.append(
        loss_mg_eucledian(x, y, encoder_weights, decoder_weights, reg)
    )
    values.append(loss_l1(x, y, encoder_weights, decoder_weights, reg))
    values.append(loss_mg_l1(x, y, encoder_weights, decoder_weights, reg))
    values.append(loss_l0(x, y, encoder_weights, decoder_weights, reg))
    values.append(loss_mg_l0(x, y, encoder_weights, decoder_weights, reg))

    for name, value in zip(names, values):
        print(f"{name}:\t{value}\n")
