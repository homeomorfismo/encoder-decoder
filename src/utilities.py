"""
Wrappers for Optax optimizers and JAX initializers.

See https://optax.readthedocs.io/en/latest/api/optimizers.html
and https://jax.readthedocs.io/en/latest/jax.nn.html#initializers
"""

from enum import Enum
import optax
import jax.nn.initializers as jinit
from typing import Callable


class OptimizerType(Enum):
    ADADELTA = optax.adadelta
    ADAFACTOR = optax.adafactor
    ADAGRAD = optax.adagrad
    ADABELIEF = optax.adabelief
    ADAN = optax.adan
    ADAM = optax.adam
    ADAMW = optax.adamw
    ADAMAX = optax.adamax
    ADAMAXW = optax.adamaxw
    FROMAGE = optax.fromage
    LAMB = optax.lamb
    LARS = optax.lars
    LBFGS = optax.lbfgs
    LION = optax.lion
    NADAM = optax.nadam
    NADAMW = optax.nadamw
    NOISY_SGD = optax.noisy_sgd
    NOVOGRAD = optax.novograd
    OPTIMISTIC_GRADIENT_DESCENT = optax.optimistic_gradient_descent
    POLYAK_SGD = optax.polyak_sgd
    RADAM = optax.radam
    RMSPROP = optax.rmsprop
    SGD = optax.sgd
    SIGN_SGD = optax.sign_sgd
    SM3 = optax.sm3
    YOGI = optax.yogi


class InitializerType(Enum):
    CONSTANT = jinit.constant
    DELTA_ORTHOGONAL = jinit.delta_orthogonal
    GLOROT_NORMAL = jinit.glorot_normal
    GLOROT_UNIFORM = jinit.glorot_uniform
    HE_NORMAL = jinit.he_normal
    HE_UNIFORM = jinit.he_uniform
    LECUN_NORMAL = jinit.lecun_normal
    LECUN_UNIFORM = jinit.lecun_uniform
    NORMAL = jinit.normal
    ONES = jinit.ones
    ORTHOGONAL = jinit.orthogonal
    TRUNCATED_NORMAL = jinit.truncated_normal
    UNIFORM = jinit.uniform
    VARIANCE_SCALING = jinit.variance_scaling
    ZEROS = jinit.zeros


def get_optimizer(optimizer_type: OptimizerType, *args, **kwargs) -> Callable:
    """
    Get the optimizer function from Optax based on the specified type.

    Args:
    - optimizer_type (OptimizerType): The type of optimizer to use.
    - *args: Positional arguments to pass to the optimizer function.
    - **kwargs: Keyword arguments to pass to the optimizer function.

    Returns:
    - Callable: The Optax optimizer function.
    """
    return optimizer_type(*args, **kwargs)


def get_initializer(
    initializer_type: InitializerType, *args, **kwargs
) -> Callable:
    """
    Get the initializer function from JAX based on the specified type.

    Args:
    - initializer_type (InitializerType): The type of initializer to use.
    - *args: Positional arguments to pass to the initializer function.
    - **kwargs: Keyword arguments to pass to the initializer function.

    Returns:
    - Callable: The JAX initializer function.
    """
    return initializer_type(*args, **kwargs)


# Example usage
if __name__ == "__main__":
    optimizer = get_optimizer(OptimizerType.ADAM, learning_rate=0.001)
    print(optimizer)

    initializer = get_initializer(InitializerType.GLOROT_UNIFORM)
    print(initializer)
