"""
Implementation of loss functions for encoder-decoder architectures
using JAX.
"""

# Note: Enum and get_loss_function at the end of the file
import jax
import jax.numpy as jnp
from jax import jit
from enum import Enum
from typing import Callable


######################
# L2 Loss functions
######################
@jit
def loss_euclidean(
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
def loss_mg_euclidean(
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


class LossFunctionType(Enum):
    EUCLIDEAN = loss_euclidean
    MG_EUCLIDEAN = loss_mg_euclidean
    L1 = loss_l1
    MG_L1 = loss_mg_l1
    L0 = loss_l0
    MG_L0 = loss_mg_l0


def get_loss_function(loss_function_type: LossFunctionType) -> Callable:
    """
    Get the loss function based on the specified type.

    Args:
    - loss_function_type (LossFunctionType): The type of loss function to use.

    Returns:
    - Callable: The JAX loss function.
    """
    return loss_function_type.value


def __test_loss_functions():
    import numpy as np

    x = np.random.rand(10, 10)
    y = np.random.rand(10, 10)
    encoder_weights = np.random.rand(10, 10)
    decoder_weights = np.random.rand(10, 10)
    reg = 0.1

    names = [
        "EUCLIDEAN",
        "MG_EUCLIDEAN",
        "L1",
        "MG_L1",
        "L0",
        "MG_L0",
    ]
    values = []
    values.append(loss_euclidean(x, y, encoder_weights, decoder_weights, reg))
    values.append(
        loss_mg_euclidean(x, y, encoder_weights, decoder_weights, reg)
    )
    values.append(loss_l1(x, y, encoder_weights, decoder_weights, reg))
    values.append(loss_mg_l1(x, y, encoder_weights, decoder_weights, reg))
    values.append(loss_l0(x, y, encoder_weights, decoder_weights, reg))
    values.append(loss_mg_l0(x, y, encoder_weights, decoder_weights, reg))

    for name, value in zip(names, values):
        print(f"{name}:\t{value}\n")


def __test_grads_loss_functions():
    import numpy as np

    x = np.random.rand(10, 10)
    y = np.random.rand(10, 10)
    encoder_weights = np.random.rand(10, 10)
    decoder_weights = np.random.rand(10, 10)
    reg = 0.1

    names = [
        "EUCLIDEAN",
        "MG_EUCLIDEAN",
        "L1",
        "MG_L1",
        "L0",
        "MG_L0",
    ]
    values = []
    values.append(
        jax.grad(loss_euclidean)(x, y, encoder_weights, decoder_weights, reg)
    )
    values.append(
        jax.grad(loss_mg_euclidean)(
            x, y, encoder_weights, decoder_weights, reg
        )
    )
    values.append(
        jax.grad(loss_l1)(x, y, encoder_weights, decoder_weights, reg)
    )
    values.append(
        jax.grad(loss_mg_l1)(x, y, encoder_weights, decoder_weights, reg)
    )
    values.append(
        jax.grad(loss_l0)(x, y, encoder_weights, decoder_weights, reg)
    )
    values.append(
        jax.grad(loss_mg_l0)(x, y, encoder_weights, decoder_weights, reg)
    )

    for name, value in zip(names, values):
        print(f"{name}:\t{value}\n")


if __name__ == "__main__":
    __test_loss_functions()
    __test_grads_loss_functions()
