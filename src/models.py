"""
Impelemntation of basic linear models for encoder-decoder architectures
in the context of multilevel methods.
Using JAX for the implementation.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap, grad, jit
import typing as tp

__NUM_EPOCHS__ = 100


class DenseEncoderDecoder:
    """
    Encoder-Decoder architecture using dense layers.
    """

    def __init__(
        self,
        fine_dim: int,
        coarse_dim: int,
    ):
        """
        Initialize the encoder-decoder architecture.

        Args:
            fine_dim: int
                Dimension of the fine grid.
            coarse_dim: int
                Dimension of the coarse grid.
        """
        self.fine_dim = fine_dim
        self.coarse_dim = coarse_dim

        self.__init_weights()

    # TODO: Use enum for different initialization methods
    def __init_weights(self):
        """
        Initialize the weights of the network.
        """
        key = random.PRNGKey(0)
        self.encoder_weights = random.normal(
            key, (self.fine_dim, self.coarse_dim)
        )
        self.decoder_weights = random.normal(
            key, (self.coarse_dim, self.fine_dim)
        )

    def encoder(
        self,
        x: jnp.ndarray,
        encoder_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Encoder function.

        Args:
            x: jnp.ndarray
                Input data.

        Returns:
            jnp.ndarray
                Encoded data.
        """
        return jnp.dot(x, encoder_weights)

    def decoder(
        self,
        x: jnp.ndarray,
        decoder_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Decoder function.

        Args:
            x: jnp.ndarray
                Input data.

        Returns:
            jnp.ndarray
                Decoded data.
        """
        return jnp.dot(x, decoder_weights)

    def __call__(
        self,
        x: jnp.ndarray,
        encoder_weights: jnp.ndarray,
        decoder_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Forward pass of the network.

        Args:
            x: jnp.ndarray
                Input data.

        Returns:
            jnp.ndarray
                Output data.
        """
        return self.decoder(self.encoder(x, encoder_weights), decoder_weights)

    def loss(
        self,
        x: jnp.ndarray,
        encoder_weights: jnp.ndarray,
        decoder_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Loss function.
        MSE + norm of the weights
        + norm of the product of the weights minus the identity matrix.

        Args:
            x: jnp.ndarray
                Input data.

        Returns:
            jnp.ndarray
                Loss value.
        """
        return (
            jnp.mean((x - self(x, encoder_weights, decoder_weights)) ** 2)
            + jnp.linalg.norm(encoder_weights)
            + jnp.linalg.norm(decoder_weights)
            + jnp.linalg.norm(
                jnp.eye(self.coarse_dim)
                - jnp.dot(decoder_weights, encoder_weights)
            )
        )

    def update(self, x: jnp.ndarray, lr: float = 0.01) -> None:
        """
        Update the weights of the network.

        Args:
            x: jnp.ndarray
                Input data.
            lr: float
                Learning rate.
        """
        loss, (grad_encoder, grad_decoder) = jax.value_and_grad(
            self.loss, (1, 2)
        )(x, self.encoder_weights, self.decoder_weights)
        self.encoder_weights -= lr * grad_encoder
        self.decoder_weights -= lr * grad_decoder


if __name__ == "__main__":
    # Test the implementation
    encoder_decoder = DenseEncoderDecoder(10, 5)
    x = jnp.ones((10, 10))
    print(
        f"Initial loss: {encoder_decoder.loss(x, encoder_decoder.encoder_weights, encoder_decoder.decoder_weights)}\n",
        f"Initial output: {encoder_decoder(x, encoder_decoder.encoder_weights, encoder_decoder.decoder_weights)}\n",
    )
    for epoch in range(__NUM_EPOCHS__):
        encoder_decoder.update(x)
        if epoch % 10 == 0:
            print(
                f"\tEpoch: {epoch}/{__NUM_EPOCHS__}\n",
                f"\tLoss: {encoder_decoder.loss(x, encoder_decoder.encoder_weights, encoder_decoder.decoder_weights)}\n",
                f"\tOutput: {encoder_decoder(x, encoder_decoder.encoder_weights, encoder_decoder.decoder_weights)}\n",
            )
    print(
        f"Final loss: {encoder_decoder.loss(x, encoder_decoder.encoder_weights, encoder_decoder.decoder_weights)}\n",
        f"Final output: {encoder_decoder(x, encoder_decoder.encoder_weights, encoder_decoder.decoder_weights)}\n",
    )
