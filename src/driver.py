"""
Driver script for testing the encoder-decoder and two-level solver models.
"""

import argparse
import toml
import ngsolve as ng
import numpy as np
import scipy as sp
import plotly.graph_objects as go
import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
from typing import Tuple, Callable, Any, Dict, List

# local imports
import utilities as ut  # optimizers, initializers
import data_gen as dg
from geo2d import make_unit_square
import loss as fn
import models as mdl
import solver as slv

# parameters
__STRICT_ATOL__: float = 1e-1
__STRICT_RTOL__: float = 1e-1


@dataclass
class MeshConfig:
    maxh: float


@dataclass
class DataGenConfig:
    tol: float
    iterations: int
    is_complex: bool
    n_samples: int
    use_restricted_operator: bool


@dataclass
class ModelConfig:
    compression_factor: float
    seed: int
    init_encoder_type: str
    init_decoder_type: str
    init_encoder_kwargs: Dict[str, Any]
    init_decoder_kwargs: Dict[str, Any]


@dataclass
class OptimizationConfig:
    optimizer_type: str
    optimizer_kwargs: Dict[str, Any]
    ord: int
    reg: float


@dataclass
class TrainingConfig:
    n_epochs: int
    freq: int
    batch_size: int


@dataclass
class CoarseningConfig:
    coarsening_type: str
    use_restricted_operator: bool
    regularization: float


@dataclass
class SolverConfig:
    solver_tol: float
    solver_max_iter: int


@dataclass
class SmootherConfig:
    smoother_tol: float
    smoother_max_iter: int


@dataclass
class OutputConfig:
    save_weights: bool
    plot_weights: bool
    strict_assert: bool


@dataclass
class Config:
    mesh: MeshConfig
    data_gen: DataGenConfig
    model: ModelConfig
    optimization: OptimizationConfig
    training: TrainingConfig
    coarsening: CoarseningConfig
    solver: SolverConfig
    smoother: SmootherConfig
    output: OutputConfig


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


def __parse_config__(config_path: str) -> Config:
    """
    Parse the TOML configuration file.
    """
    with open(config_path, "r") as f:
        config_dict = toml.load(f)
    __assert_minimal_config__(config_dict)
    return Config(
        mesh=MeshConfig(**config_dict["mesh"]),
        data_gen=DataGenConfig(**config_dict["data_gen"]),
        model=ModelConfig(**config_dict["model"]),
        optimization=OptimizationConfig(**config_dict["optimization"]),
        training=TrainingConfig(**config_dict["training"]),
        coarsening=CoarseningConfig(**config_dict["coarsening"]),
        solver=SolverConfig(**config_dict["solver"]),
        smoother=SmootherConfig(**config_dict["smoother"]),
        output=OutputConfig(**config_dict["output"]),
    )


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
        "coarsening",
        "solver",
        "smoother",
        "output",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")


def linear_encoder_decoder(config: Config) -> None:
    """
    Train a linear encoder-decoder model.
    Check the configuration file for more details.
    """
    maxh = config.mesh.maxh
    dg_tol = config.data_gen.tol
    dg_iterations = config.data_gen.iterations
    is_complex = config.data_gen.is_complex
    n_samples = config.data_gen.n_samples
    use_restricted_operator = config.data_gen.use_restricted_operator
    compression_factor = config.model.compression_factor
    seed = config.model.seed
    init_encoder_type = config.model.init_encoder_type
    init_decoder_type = config.model.init_decoder_type
    init_encoder_kwargs = config.model.init_encoder_kwargs
    init_decoder_kwargs = config.model.init_decoder_kwargs
    optimizer_type = config.optimization.optimizer_type
    optimizer_kwargs = config.optimization.optimizer_kwargs
    ord = config.optimization.ord
    reg = config.optimization.reg
    n_epochs = config.training.n_epochs
    freq = config.training.freq
    batch_size = config.training.batch_size
    coarsening_type = config.coarsening.coarsening_type
    use_restricted_operator = config.coarsening.use_restricted_operator
    regularization = config.coarsening.regularization
    solver_tol = config.solver.solver_tol
    solver_max_iter = config.solver.solver_max_iter
    smoother_tol = config.smoother.smoother_tol
    smoother_max_iter = config.smoother.smoother_max_iter
    save_weights = config.output.save_weights
    plot_weights = config.output.plot_weights
    strict_assert = config.output.strict_assert

    print("\n->Creating mesh and data...")
    square = ng.Mesh(make_unit_square().GenerateMesh(maxh=maxh))
    conv_diff_dgen = dg.BasicConvDiffDataGen(
        square,
        tol=dg_tol,
        iterations=dg_iterations,
        is_complex=is_complex,
    )
    x_data = conv_diff_dgen.generate_samples(
        n_samples, use_rest=use_restricted_operator
    )
    print(
        f"\n\t-> Generated {n_samples} samples"
        f"\n\t-> Using restricted operator in DataGen: {use_restricted_operator}"
        "\n->Creating encoder-decoder weights..."
    )

    n_fine = conv_diff_dgen.space.ndof
    n_coarse = int(n_fine // compression_factor)
    jax_key = jax.random.PRNGKey(seed)
    jax_type = jnp.complex128 if is_complex else jnp.float64

    init_encoder = ut.get_initializer(init_encoder_type, **init_encoder_kwargs)
    init_decoder = ut.get_initializer(init_decoder_type, **init_decoder_kwargs)

    jax_key, k1, k2 = jax.random.split(jax_key, 3)
    weights_encoder = init_encoder(k1, (n_fine, n_coarse), dtype=jax_type)
    weights_decoder = init_decoder(k2, (n_coarse, n_fine), dtype=jax_type)
    params = (weights_encoder, weights_decoder)
    del k1, k2
    print("\n->Training the encoder-decoder model...")

    optimizer = ut.get_optimizer(optimizer_type, **optimizer_kwargs)
    optimizer_state = optimizer.init(params)
    loss_fn = fn.get_loss(ord)
    print(
        f"\n\t-> Using optimizer: {optimizer_type}"
        f"\n\t-> Using loss function regularized with L{ord}-norm and strength {reg}"
    )

    @jax.jit
    def update_step(carry, batch):
        param, opt_state = carry
        loss, grads = jax.value_and_grad(
            loss_fn,
            argnums=(1, 2),
            holomorphic=is_complex,
        )(batch, *param, reg)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_param = optax.apply_updates(param, updates)
        return (new_param, new_opt_state), loss

    n_samples = x_data.shape[0]
    n_batches = n_samples // batch_size

    # x_data.shape = (n_samples, n_fine)
    for epoch in range(n_epochs):
        jax_key, k1 = jax.random.split(jax_key)
        permutation = jax.random.permutation(k1, n_samples)
        del k1
        x_data = x_data[permutation]

        batch_data = x_data[: n_batches * batch_size].reshape(
            (n_batches, batch_size, -1)
        )

        carry = (weights_encoder, weights_decoder), optimizer_state
        carry, loss = jax.lax.scan(update_step, carry, batch_data)
        (weights_encoder, weights_decoder), optimizer_state = carry

        mean_loss = jnp.mean(loss)
        if epoch % freq == 0 or epoch == n_epochs - 1:
            print(
                f"\n\t-> Epoch {epoch+1}/{n_epochs}"
                f"\n\t-> (Mean) Loss: {mean_loss:.10f}"
            )
        if strict_assert:
            assert not jnp.isnan(loss).any(), "Loss is NaN!"

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

    jax_grid_fun = jnp.array(grid_fun.vec.FV().NumPy(), dtype=jax_type)
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
    if strict_assert:
        assert np.isclose(
            error,
            0.0,
            atol=__STRICT_ATOL__,
            rtol=__STRICT_RTOL__,
        ), f"L2 error: {error:.10f}, expected less than {__STRICT_ATOL__}!"

    # Test 2: Wrap the encoder-decoder model in a two-level solver
    fine_operator = conv_diff_dgen.rest_operator
    if use_restricted_operator:
        fine_operator_loc = fine_operator
    else:
        fine_operator_loc = conv_diff_dgen.operator

    # dec-dec, dec-enc, enc-enc, enc-dec
    if coarsening_type == "enc-dec":
        fine_to_coarse = weights_encoder
        coarse_to_fine = weights_decoder
    elif coarsening_type == "dec-enc":
        fine_to_coarse = weights_decoder.T
        coarse_to_fine = weights_encoder.T
    elif coarsening_type == "enc-enc":
        fine_to_coarse = weights_encoder.T
        coarse_to_fine = weights_encoder
    elif coarsening_type == "dec-dec":
        fine_to_coarse = weights_decoder.T
        coarse_to_fine = weights_decoder
    else:
        raise ValueError(f"Invalid coarsening type: {coarsening_type}")

    print(f"\t-> Coarsening type: {coarsening_type}")
    coarse_operator = jnp.dot(
        jnp.dot(fine_to_coarse, fine_operator_loc), coarse_to_fine
    )

    if regularization > 0.0:
        print(
            f"\n\t-> Regularizing the coarse operator with strength {regularization}"
        )
        coarse_operator += regularization * jnp.eye(n_coarse)

    # Call the two-level solver
    print(
        "\n->Testing the two-level solver model..."
        "\n-> - Lap(u) + u = f in [0,1]^2"
        "\n\t-> Solving u(x,y) = x * (1 - x) * y * (1 - y)"
        "\n\t-> RHS f(x,y) = 2( x * (1 - x) + y * (1 - y) )"
        " + u(x,y) in [0,1]^2"
    )
    rhs_grid_fun = conv_diff_dgen.get_gf(name="rhs")
    rhs_grid_fun.Set(
        2 * (ng.x * (1 - ng.x) + ng.y * (1 - ng.y))
        + ng.x * (1 - ng.x) * ng.y * (1 - ng.y)
    )
    rhs = jnp.array(rhs_grid_fun.vec.FV().NumPy(), dtype=jax_type)

    jax_reconstr = slv.encoder_decoder_tl(
        fine_operator,
        coarse_operator,
        fine_to_coarse.T,
        coarse_to_fine.T,
        rhs,
        solver_tol=solver_tol,
        solver_max_iter=solver_max_iter,
        smoother_tol=smoother_tol,
        smoother_max_iter=smoother_max_iter,
    )

    solution = conv_diff_dgen.get_gf(name="TL(x * (1 - x) * y * (1 - y))")
    solution.vec.FV().NumPy()[:] = jax_reconstr

    # Compute NGSolve L2 error, assert close to zero
    exact_grid_fun = conv_diff_dgen.get_gf(name="u(x,y)")
    exact_grid_fun.Set(ng.x * (1 - ng.x) * ng.y * (1 - ng.y))

    ng.Draw(rhs_grid_fun, mesh=square, name="rhs")
    ng.Draw(exact_grid_fun, mesh=square, name="u(x,y)")
    ng.Draw(solution, mesh=square, name="TL(x * (1 - x) * y * (1 - y))")

    error = ng.sqrt(
        ng.Integrate(
            ng.InnerProduct(
                solution - exact_grid_fun, solution - exact_grid_fun
            )
            * ng.dx,
            square,
        )
    )
    print(f"\n\t-> L2 error: {error:.10f}")
    if strict_assert:
        assert np.isclose(
            error,
            0.0,
            atol=__STRICT_ATOL__,
            rtol=__STRICT_RTOL__,
        ), f"Error: {error:.10f}, expected less than 1e-1!"

    print("\n->Done!")


if __name__ == "__main__":
    # args = __parse_args__()
    # config = __parse_config__(args.config)
    config = __parse_config__("default.toml")  # use with NGSolve
    linear_encoder_decoder(config)
