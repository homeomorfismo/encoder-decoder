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
import ngsolve as ng

# local imports
import utilities as ut  # optimizers, initializers
from loss import get_loss, get_mg_loss
from models import LinearEncoderDecoder, MGLinearEncoderDecoder
from geo2d import make_unit_square
from data_gen import BasicConvDiffDataGen

# Parameters
__DIM__: int = 10
__REG__: float = 0.1
__ORD_TYPES__: list = [0, 1, 2, jnp.inf, -jnp.inf, "fro", "nuc"]

__SEED__: int = 0
__NUM_EPOCHS__: int = 100
__BATCH_SIZE__: int = 100
__LEARNING_RATE__: float = 0.01
__OPTIMIZER_NAME__: str = "adam"
__INIT_NAME__: str = "glorot_uniform"

__MAXH__: float = 0.1
__SOLVER_TOL__: float = 1e-1
__ORDER__: int = 1
__SOLVER_ITER__: int = 5
__NUM_SAMPLES__: int = 8


def __get_mnist_data():
    """
    Get the MNIST data.
    """
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = jnp.array(x_train.reshape(-1, 784).astype("float32") / 255.0)
    x_test = jnp.array(x_test.reshape(-1, 784).astype("float32") / 255.0)
    return x_train, x_test


# Testing loss.py
def test_get_loss():
    """
    Test all the loss functions.
    """
    x = np.random.rand(__DIM__, __DIM__)
    encoder_weights = np.random.rand(__DIM__, __DIM__)
    decoder_weights = np.random.rand(__DIM__, __DIM__)
    range_weights = np.random.rand(__DIM__, __DIM__)
    for ord in __ORD_TYPES__:
        loss_fn = get_loss(ord)
        mg_loss_fn = get_mg_loss(ord)
        loss_fn_val = loss_fn(x, encoder_weights, decoder_weights, __REG__)
        mg_loss_fn_val = mg_loss_fn(
            x,
            encoder_weights,
            decoder_weights,
            range_weights,
            __REG__,
        )
        print(
            f"\nLoss function with ord={ord}:"
            f"\nLoss: {loss_fn_val}"
            f"\nMG Loss: {mg_loss_fn_val}"
        )


# Testing models.py
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


# Testing data_gen.py
def test_basic_conv_diff_data_gen():
    """
    Test the BasicConvDiffDataGen function.
    """
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=__MAXH__))
    data_gen = BasicConvDiffDataGen(
        mesh,
        tol=__SOLVER_TOL__,
        order=__ORDER__,
        iterations=__SOLVER_ITER__,
        is_complex=True,
        is_dirichlet=True,
        debug=True,
    )

    # rnd_shape test
    key = jax.random.PRNGKey(__SEED__)
    key, rnd = data_gen.rnd_shape(key, ())
    print(f"Key: {key}, Rnd: {rnd}")

    # generate_samples test
    samples_full_op = data_gen.generate_samples(
        __NUM_SAMPLES__, use_rest=False
    )
    samples_rest_op = data_gen.generate_samples(__NUM_SAMPLES__, use_rest=True)
    gfs = data_gen.get_gf(dim=__NUM_SAMPLES__ * 2)

    for i in range(__NUM_SAMPLES__):
        gfs.vecs[i].FV().NumPy()[:] = samples_full_op[i]
        gfs.vecs[i + __NUM_SAMPLES__].FV().NumPy()[:] = samples_rest_op[i]

    ng.Draw(mesh)
    ng.Draw(gfs)


if __name__ == "__main__":
    test_get_loss()
    test_linear_encoder_decoder()
    test_mg_linear_encoder_decoder()
    test_basic_conv_diff_data_gen()
