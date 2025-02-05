"""
Testing module for the encoder-decoder architecture.
"""

import jax
from jax import jit
import jax.numpy as jnp
import jax.lax as lax
import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np

from models import LinearEncoderDecoder, MGLinearEncoderDecoder
import utilities as ut  # optimizers, initializers

__SEED__: int = 0
__NUM_EPOCHS__: int = 100
__BATCH_SIZE__: int = 100
__LEARNING_RATE__: float = 0.01
__OPTIMIZER_NAME__: str = "adam"
__INIT_NAME__: str = "glorot_uniform"


def __get_mnist_data():
    """
    Get the MNIST data.
    """
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = jnp.array(x_train.reshape(-1, 784).astype("float32") / 255.0)
    x_test = jnp.array(x_test.reshape(-1, 784).astype("float32") / 255.0)
    return x_train, x_test


def test_linear_encoder_decoder():
    """
    Test the LinearEncoderDecoder function.
    """
    x_train, x_test = __get_mnist_data()

    fine_dim: int = x_train.shape[1]
    coarse_dim: int = 100

    initializers = ut.get_initializer(__INIT_NAME__)
    key = jax.random.PRNGKey(__SEED__)
    key1, key2 = jax.random.split(key)
    encoder_weights = initializers(key1, (fine_dim, coarse_dim))
    decoder_weights = initializers(key2, (coarse_dim, fine_dim))
    params = (encoder_weights, decoder_weights)

    optimizer = ut.get_optimizer(__OPTIMIZER_NAME__, __LEARNING_RATE__)
    opt_state = optimizer.init(params)

    n_samples = x_train.shape[0]
    n_batches = n_samples // __BATCH_SIZE__

    # TODO: Replace with personalized loss function
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

    for epoch in range(__NUM_EPOCHS__):
        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(subkey, n_samples)
        shuffled_x = x_train[permutation]

        batched_data = shuffled_x[: n_batches * __BATCH_SIZE__].reshape(
            (n_batches, __BATCH_SIZE__, -1)
        )

        (params, opt_state), losses = lax.scan(
            update_step, (params, opt_state), batched_data
        )

        mean_loss = jnp.mean(losses)
        print(f"Epoch {epoch + 1}, Loss: {mean_loss:.4f}")

    test_loss = loss_fn(params, x_test)
    print(f"Test Loss: {test_loss:.4f}")

    # Visualize the results
    n_samples = 5
    indices = np.random.choice(x_test.shape[0], n_samples)
    x_samples = x_test[indices]
    y_samples = LinearEncoderDecoder(x_samples, *params)

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 10))
    for i in range(n_samples):
        axes[i, 0].imshow(x_samples[i].reshape(28, 28), cmap="gray")
        axes[i, 1].imshow(y_samples[i].reshape(28, 28), cmap="gray")
    plt.show()


def test_mg_linear_encoder_decoder():
    """
    Test the MGLinearEncoderDecoder function.
    """
    x_train, x_test = __get_mnist_data()

    fine_dim: int = x_train.shape[1]
    coarse_dim: int = 100

    initializers = ut.get_initializer(__INIT_NAME__)
    key = jax.random.PRNGKey(__SEED__)
    key1, key2 = jax.random.split(key)
    encoder_weights = initializers(key1, (fine_dim, coarse_dim))
    decoder_weights = initializers(key2, (coarse_dim, fine_dim))
    id_matrix = jnp.eye(fine_dim)

    params = (encoder_weights, decoder_weights)

    optimizer = ut.get_optimizer(__OPTIMIZER_NAME__, __LEARNING_RATE__)
    opt_state = optimizer.init(params)

    n_samples = x_train.shape[0]
    n_batches = n_samples // __BATCH_SIZE__

    # TODO: Replace with personalized loss function
    def loss_fn(params, x):
        return 0.5 * jnp.mean(
            jnp.square(x - MGLinearEncoderDecoder(x, *params, id_matrix))
        )

    @jit
    def update_step(carry, batch):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = jax.tree_util.tree_map(
            lambda p, u: p + u, params, updates
        )
        return (new_params, new_opt_state), loss

    for epoch in range(__NUM_EPOCHS__):
        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(subkey, n_samples)
        shuffled_x = x_train[permutation]

        batched_data = shuffled_x[: n_batches * __BATCH_SIZE__].reshape(
            (n_batches, __BATCH_SIZE__, -1)
        )

        (params, opt_state), losses = lax.scan(
            update_step, (params, opt_state), batched_data
        )

        mean_loss = jnp.mean(losses)
        print(f"Epoch {epoch + 1}, Loss: {mean_loss:.4f}")

    test_loss = loss_fn(params, x_test)
    print(f"Test Loss: {test_loss:.4f}")

    # Visualize the results
    n_samples = 5
    indices = np.random.choice(x_test.shape[0], n_samples)
    x_samples = x_test[indices]
    y_samples = MGLinearEncoderDecoder(x_samples, *params, id_matrix)

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 10))
    for i in range(n_samples):
        axes[i, 0].imshow(x_samples[i].reshape(28, 28), cmap="gray")
        axes[i, 1].imshow(y_samples[i].reshape(28, 28), cmap="gray")
    plt.show()
