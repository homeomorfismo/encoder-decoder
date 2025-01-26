"""
Wrappers for testing the encoder-decoder model and the two-level solver models,
using JAX.
"""

import argparse
import toml

import ngsolve as ng
import numpy as np
import scipy as sp
import plotly.graph_objects as go
import jax
import jax.numpy as jnp
import jax.nn.initializers as jinit
import optax

# local imports
import utilities as ut  # optimizers, initializers
import data_gen as dg
from geo2d import make_unit_square
import loss as fn
import models as mdl
import solver as slv
import sparse as sps


def __parse_args__() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Test the encoder-decoder and two-level solver models."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the TOML configuration file.",
    )
    args = parser.parse_args()
    return args


def __parse_config__(config_path: str) -> dict:
    """
    Parse the TOML configuration file.
    """
    with open(config_path, "r") as f:
        config = toml.load(f)
    __assert_minimal_config__(config)
    return config


def __assert_minimal_config__(config: dict) -> None:
    """
    Assert that the minimal configuration is present.
    """
    required_keys = [
        "mesh",
        "data_gen",
        "model",
        "optimization",
        "training",
        "output",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")


def linear_encoder_decoder(config: dict) -> None:
    """
    Train a linear encoder-decoder model.
    Check the configuration file for more details.
    """
    maxh = config["mesh"]["maxh"]
    tol = config["data_gen"]["tol"]
    iterations = config["data_gen"]["iterations"]
    is_complex = config["data_gen"]["is_complex"]
    n_samples = config["data_gen"]["n_samples"]
    compression_factor = config["model"]["compression_factor"]
    init_encoder_type = config["model"]["init_encoder_type"]
    init_decoder_type = config["model"]["init_decoder_type"]
    init_encoder_kwargs = config["model"]["init_encoder_kwargs"]
    init_decoder_kwargs = config["model"]["init_decoder_kwargs"]
    optimizer_type = config["optimization"]["optimizer_type"]
    optimizer_kwargs = config["optimization"]["optimizer_kwargs"]
    ord = config["optimization"]["ord"]
    reg = config["optimization"]["reg"]
    n_epochs = config["training"]["n_epochs"]
    freq = config["training"]["freq"]
    save_weights = config["output"]["save_weights"]
    plot_weights = config["output"]["plot_weights"]

    print("\n->Creating mesh and data...")
    square = ng.Mesh(make_unit_square().GenerateMesh(maxh=maxh))
    conv_diff_dgen = dg.BasicConvDiffDataGen(
        square,
        tol=tol,
        iterations=iterations,
        is_complex=is_complex,
    )
    x_data = conv_diff_dgen.generate_samples(n_samples)

    print("\n->Creating encoder-decoder weights...")

    n_fine = conv_diff_dgen.space.ndof
    n_coarse = int(n_fine // compression_factor)
    jax_key = jax.random.PRNGKey(0)
    jax_type = jnp.complex64 if is_complex else jnp.float32

    init_encoder = ut.get_initializer(init_encoder_type, **init_encoder_kwargs)
    init_decoder = ut.get_initializer(init_decoder_type, **init_decoder_kwargs)

    # Note: this fully defines the encoder-decoder model
    weights_encoder = init_encoder(jax_key, (n_fine, n_coarse), dtype=jax_type)
    weights_decoder = init_decoder(jax_key, (n_coarse, n_fine), dtype=jax_type)

    print("\n->Training the encoder-decoder model...")

    # Define an optimizer
    print(f"\n\t-> Using optimizer: {optimizer_type}")
    print(
        f"\n\t-> Using loss function regularized with L{ord}-norm and strength {reg}"
    )
    optimizer = ut.get_optimizer(optimizer_type, **optimizer_kwargs)
    optimizer_state = optimizer.init((weights_encoder, weights_decoder))

    # Define the loss function
    loss_fn = fn.get_loss(ord)

    # Update the weights
    for epoch in range(n_epochs):
        # Compute the gradients
        val_loss, grads = jax.value_and_grad(
            loss_fn,
            argnums=(2, 3),
            holomorphic=is_complex,
        )(x_data, x_data, weights_encoder, weights_decoder, reg)

        # Update the weights
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        weights_encoder, weights_decoder = optax.apply_updates(
            (weights_encoder, weights_decoder), updates
        )
        if epoch % freq == 0 or epoch == n_epochs - 1:
            print(
                f"\n\t-> Epoch {epoch+1}/{n_epochs}"
                f"\n\t-> Loss: {val_loss:.10f}"
            )

    # Save the weights
    if save_weights:
        print("\n->Saving the encoder-decoder weights...")
        weights_encoder = jax.device_get(weights_encoder)
        weights_decoder = jax.device_get(weights_decoder)
        np.save("weights_encoder.npy", weights_encoder)
        np.save("weights_decoder.npy", weights_decoder)

    # Plot heatmaps of the sparsity patterns of the weights
    if plot_weights:
        print("\n->Plotting the encoder-decoder weights...")
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=np.abs(weights_encoder),
                colorscale="Viridis",
                colorbar=dict(title="Encoder Weights"),
            )
        )
        fig.update_layout(title="Encoder-Decoder Weights")
        fig.show()

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=np.abs(weights_decoder),
                colorscale="Viridis",
                colorbar=dict(title="Decoder Weights"),
            )
        )
        fig.update_layout(title="Encoder-Decoder Weights")
        fig.show()

    print("\n->Testing the encoder-decoder model...")

    # Test 1: Encode-decode a simple sinusoidal function
    print(
        "\n\t-> Testing the encoder-decoder model on a simple sinusoidal function..."
    )
    grid_fun = conv_diff_dgen.get_gf(name="cos(xy) + sin(xy)")
    reconstr = conv_diff_dgen.get_gf(name="ED(cos(xy) + sin(xy))")
    grid_fun.Set(
        ng.cos(ng.x * ng.y) + ng.sin(ng.x * ng.y)
    )  # cos(xy) + sin(xy)
    ng.Draw(grid_fun, mesh=square, name="cos(xy) + sin(xy)")

    jax_grid_fun = jnp.array(
        grid_fun.vec.FV().NumPy(), dtype=jax_type
    ).flatten()
    jax_reconstr = mdl.LinearEncoderDecoder(
        jax_grid_fun, weights_encoder, weights_decoder
    )
    reconstr.vec.FV().NumPy()[:] = jax_reconstr
    ng.Draw(reconstr, mesh=square, name="ED(cos(xy) + sin(xy))")

    # Compute NGSolve L2 error, assert close to zero
    error = ng.sqrt(
        ng.Integrate(
            ng.InnerProduct(grid_fun - reconstr, grid_fun - reconstr) * ng.dx,
            square,
        )
    )
    print(f"\n\t-> L2 error: {error:.10f}")
    # assert np.isclose(error, 0.0, atol=1e-1), f"Error: {error:.10f}, expected less than 1e-1!"

    # Test 2: Wrap the encoder-decoder model in a two-level solver
    # TODO
    pass


if __name__ == "__main__":
    # args = __parse_args__()
    # config = __parse_config__(args.config)
    config = __parse_config__("default.toml")  # use with NGSolve
    linear_encoder_decoder(config)
