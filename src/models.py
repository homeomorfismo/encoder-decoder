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


class DenseEncoderDecoder:
    """
    Encoder-Decoder architecture using dense layers.
    """

    def __init__(self, fine_dim: int, coarse_dim: int):
        self.fine_dim = fine_dim
        self.coarse_dim = coarse_dim
        self.key = random.PRNGKey(0)
        self.encoder_weights = random.normal(self.key, (fine_dim, coarse_dim))
        self.decoder_weights = random.normal(self.key, (coarse_dim, fine_dim))

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, self.encoder_weights)

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, self.decoder_weights)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.decode(self.encode(x))


def loss(x, y, encoder_weights, decoder_weights, coarse_dim=5, alpha=1.0e-3):
    """
    Loss function for encoder-decoder architecture.
    """
    reconstruction_loss = jnp.mean((x - y) ** 2)
    regularization_loss = jnp.linalg.norm(encoder_weights)
    regularization_loss += jnp.linalg.norm(decoder_weights)
    identity_loss = jnp.linalg.norm(
        jnp.eye(coarse_dim) - jnp.dot(decoder_weights, encoder_weights)
    )
    return reconstruction_loss + alpha * (regularization_loss + identity_loss)


def update(x, encoder_decoder, lr=0.01):
    loss_value, grads = jax.value_and_grad(loss, argnums=(2, 3))(
        x,
        encoder_decoder(x),
        encoder_decoder.encoder_weights,
        encoder_decoder.decoder_weights,
    )
    encoder_decoder.encoder_weights -= lr * grads[0]
    encoder_decoder.decoder_weights -= lr * grads[1]
    return loss_value


if __name__ == "__main__":
    encoder_decoder = DenseEncoderDecoder(__FINE_DIM__, __COARSE_DIM__)
    key = random.PRNGKey(0)
    x = random.normal(key, (__BATCH_SIZE__, __FINE_DIM__))
    for epoch in range(__NUM_EPOCHS__):
        for _ in range(__NUM_BATCHES__):
            loss_value = update(x, encoder_decoder, __LEARNING_RATE__)
        if epoch % 10 == 0:
            print(f"\nEpoch: {epoch}")
            print(f"\n\tLoss: {loss_value}")
            print(f"\n\tEncoder weights: {encoder_decoder.encoder_weights}")
            print(f"\n\tDecoder weights: {encoder_decoder.decoder_weights}")
    print("\nTraining complete!\n")
