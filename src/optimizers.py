"""
Wrappers for Optax optimizers.
(This is moreso for a list/reminder of available optimizers in Optax.)
See https://optax.readthedocs.io/en/latest/api/optimizers.html
"""

from enum import Enum
import optax
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
    POLYAK_SDG = optax.polyak_sgd
    RADAM = optax.radam
    RMSPROP = optax.rmsprop
    SGD = optax.sgd
    SIGN_SGD = optax.sign_sgd
    SM3 = optax.sm3
    YOGI = optax.yogi


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


# Example usage
if __name__ == "__main__":
    optimizer = get_optimizer(OptimizerType.ADAM, learning_rate=0.001)
    print(optimizer)
