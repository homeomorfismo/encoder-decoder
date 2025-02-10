"""
Wrappers for Optax optimizers and JAX initializers.

See https://optax.readthedocs.io/en/latest/api/optimizers.html
and https://jax.readthedocs.io/en/latest/jax.nn.html#initializers
"""

import optax
import jax.nn.initializers as jinit
from typing import Callable

# Dictionary for Optax optimizers
OPTIMIZERS = {
    "adadelta": optax.adadelta,
    "adafactor": optax.adafactor,
    "adagrad": optax.adagrad,
    "adabelief": optax.adabelief,
    "adan": optax.adan,
    "adam": optax.adam,
    "adamw": optax.adamw,
    "adamax": optax.adamax,
    "adamaxw": optax.adamaxw,
    "fromage": optax.fromage,
    "lamb": optax.lamb,
    "lars": optax.lars,
    "lbfgs": optax.lbfgs,
    "lion": optax.lion,
    "nadam": optax.nadam,
    "nadamw": optax.nadamw,
    "noisy_sgd": optax.noisy_sgd,
    "novograd": optax.novograd,
    "optimistic_gradient_descent": optax.optimistic_gradient_descent,
    "polyak_sgd": optax.polyak_sgd,
    "radam": optax.radam,
    "rmsprop": optax.rmsprop,
    "sgd": optax.sgd,
    "sign_sgd": optax.sign_sgd,
    "sm3": optax.sm3,
    "yogi": optax.yogi,
}

# Dictionary for JAX initializers
INITIALIZERS = {
    "constant": jinit.constant,
    "delta_orthogonal": jinit.delta_orthogonal,
    "glorot_normal": jinit.glorot_normal,
    "glorot_uniform": jinit.glorot_uniform,
    "he_normal": jinit.he_normal,
    "he_uniform": jinit.he_uniform,
    "lecun_normal": jinit.lecun_normal,
    "lecun_uniform": jinit.lecun_uniform,
    "normal": jinit.normal,
    "ones": jinit.ones,
    "orthogonal": jinit.orthogonal,
    "truncated_normal": jinit.truncated_normal,
    "uniform": jinit.uniform,
    "variance_scaling": jinit.variance_scaling,
    "zeros": jinit.zeros,
}


def get_optimizer(optimizer_type: str, *args, **kwargs) -> Callable:
    """
    Get the optimizer function from Optax based on the specified type.

    Args:
    - optimizer_type (str): The type of optimizer to use.
    - *args: Positional arguments to pass to the optimizer function.
    - **kwargs: Keyword arguments to pass to the optimizer function.

    Returns:
    - Callable: The Optax optimizer function.
    """
    optimizer_func = OPTIMIZERS.get(optimizer_type.lower())
    if optimizer_func is None:
        raise ValueError(f"Optimizer '{optimizer_type}' not found.")
    return optimizer_func(*args, **kwargs)


def get_initializer(initializer_type: str, *args, **kwargs) -> Callable:
    """
    Get the initializer function from JAX based on the specified type.

    Args:
    - initializer_type (str): The type of initializer to use.
    - *args: Positional arguments to pass to the initializer function.
    - **kwargs: Keyword arguments to pass to the initializer function.

    Returns:
    - Callable: The JAX initializer function.
    """
    init_func = INITIALIZERS.get(initializer_type.lower())
    if init_func is None:
        raise ValueError(f"Initializer '{initializer_type}' not found.")
    return init_func(*args, **kwargs)
