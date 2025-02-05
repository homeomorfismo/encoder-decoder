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
    assert x.ndim == 2, "Input must be 2D"
    assert weights.ndim == 2, "Weights must be 2D"
    assert bias.ndim == 1, "Bias must be 1D"
    assert (
        x.shape[1] == weights.shape[0]
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
        x, encoder_weights, jnp.zeros(encoder_weights.shape[1], dtype=x.dtype)
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
        x, encoder_weights, jnp.zeros(encoder_weights.shape[1], dtype=x.dtype)
    )
    fine_x = LinearLayer(
        coarse_x,
        decoder_weights,
        jnp.zeros(decoder_weights.shape[1], dtype=x.dtype),
    )
    range_x = LinearLayer(
        fine_x, range_weights, jnp.zeros(range_weights.shape[1], dtype=x.dtype)
    )
    return range_x


if __name__ == "__main__":
    import jax.lax as lax
    import tensorflow.keras.datasets.mnist as mnist
    import utilities as ut  # optimizers, initializers

    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = jnp.array(x_train.reshape(-1, 784).astype("float32") / 255.0)
    x_test = jnp.array(x_test.reshape(-1, 784).astype("float32") / 255.0)

    fine_dim: int = x_train.shape[1]
    coarse_dim: int = 100

    # Initialize weights
    initializers = ut.get_initializer(__INIT_NAME__)
    key = random.PRNGKey(__SEED__)
    key1, key2 = random.split(key)
    encoder_weights = initializers(key1, (fine_dim, coarse_dim))
    decoder_weights = initializers(key2, (coarse_dim, fine_dim))
    params = (encoder_weights, decoder_weights)

    optimizer = ut.get_optimizer("adam", __LEARNING_RATE__)
    opt_state = optimizer.init(params)

    n_samples = x_train.shape[0]
    n_batches = n_samples // __BATCH_SIZE__

    def loss_fn(params, x):
        return 0.5 * jnp.mean(jnp.square(x - LinearEncoderDecoder(x, *params)))

    @jit
    def update_step(carry, batch):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = jax.tree_util.tree_map(
            lambda p, u: p + u, params, updates
        )
        return (new_params, new_opt_state), loss

    # Training loop
    for epoch in range(__NUM_EPOCHS__):
        key, subkey = random.split(key)
        permutation = random.permutation(subkey, n_samples)
        shuffled_x = x_train[permutation]

        # Reshape data into batches
        batched_data = shuffled_x[: n_batches * __BATCH_SIZE__].reshape(
            (n_batches, __BATCH_SIZE__, -1)
        )

        # Run training steps for one epoch
        (params, opt_state), losses = lax.scan(
            update_step, (params, opt_state), batched_data
        )

        mean_loss = jnp.mean(losses)
        print(f"Epoch {epoch + 1}, Loss: {mean_loss:.4f}")

    # Test the model
    test_loss = loss_fn(params, x_test)
    print(f"Test Loss: {test_loss:.4f}")

    # Visualize the results
    import matplotlib.pyplot as plt
    import numpy as np

    n_samples = 5
    indices = np.random.choice(x_test.shape[0], n_samples)
    x_samples = x_test[indices]
    y_samples = LinearEncoderDecoder(x_samples, *params)

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 10))
    for i in range(n_samples):
        axes[i, 0].imshow(x_samples[i].reshape(28, 28), cmap="gray")
        axes[i, 1].imshow(y_samples[i].reshape(28, 28), cmap="gray")
    plt.show()
