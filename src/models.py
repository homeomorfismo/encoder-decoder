"""
Implementation of basic linear models for encoder-decoder architectures
using JAX.
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax import random, jit

# Parameters
__NUM_EPOCHS__ = 100
__COARSE_DIM__ = 5
__FINE_DIM__ = 10
__BATCH_SIZE__ = 5
__NUM_BATCHES__ = 50
__LEARNING_RATE__ = 0.05


@jit
def LinearLayer(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    """
    Simple linear layer x -> Wx + b.
    """
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
        x, encoder_weights, jnp.zeros_like(encoder_weights.shape[1])
    )
    fine_x = LinearLayer(
        coarse_x, decoder_weights, jnp.zeros_like(decoder_weights.shape[1])
    )
    return fine_x


@jit
def loss(
    x: jnp.ndarray,
    y: jnp.ndarray,
    encoder_weights: jnp.ndarray,
    decoder_weights: jnp.ndarray,
    reg: jnp.ndarray,
):
    """
    Loss function for encoder-decoder architecture.
    """
    reconstr_loss = jnp.mean((x - y) ** 2)
    reg_loss = jnp.linalg.norm(encoder_weights)
    reg_loss += jnp.linalg.norm(decoder_weights)
    reg_loss += jnp.linalg.norm(
        jnp.eye(__COARSE_DIM__) - jnp.dot(decoder_weights, encoder_weights)
    )
    return reconstr_loss + reg * reg_loss


@jit
def update(
    x: jnp.ndarray,
    encoder_weights: jnp.ndarray,
    decoder_weights: jnp.ndarray,
    lr: float = 0.01,
    reg: float = 0.01,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Update function for training the encoder-decoder architecture.
    """
    grad_ew, grad_dw = jax.grad(loss, argnums=(2, 3))(
        x,
        LinearEncoderDecoder(x, encoder_weights, decoder_weights),
        encoder_weights,
        decoder_weights,
        jnp.array(reg),
    )
    encoder_weights -= lr * grad_ew
    decoder_weights -= lr * grad_dw
    return encoder_weights, decoder_weights


if __name__ == "__main__":
    # Initialize weights
    key = random.PRNGKey(0)
    encoder_weights = random.normal(key, (__FINE_DIM__, __COARSE_DIM__))
    decoder_weights = random.normal(key, (__COARSE_DIM__, __FINE_DIM__))

    # Training loop
    for epoch in range(__NUM_EPOCHS__):
        for _ in range(__NUM_BATCHES__):
            x = random.normal(key, (__BATCH_SIZE__, __FINE_DIM__))
            encoder_weights, decoder_weights = update(
                x, encoder_weights, decoder_weights, __LEARNING_RATE__
            )
        print(f"Epoch {epoch} completed")
        if epoch % 10 == 0:
            loss_val = loss(
                x,
                LinearEncoderDecoder(x, encoder_weights, decoder_weights),
                encoder_weights,
                decoder_weights,
                0.01,
            )
            print(
                f"\n\tEncoder weights:\n{encoder_weights}"
                f"\n\tDecoder weights:\n{decoder_weights}"
                f"\n\tLoss: {loss_val}\n"
            )

    print("Training completed")
