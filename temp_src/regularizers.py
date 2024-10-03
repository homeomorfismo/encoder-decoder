"""
This file contains custom regularizers for symmetric and identity matrices.
List of regularizers:
    L1L2ProjectionRegularization
    - L1 regularization plus L2 (symmetric) projection constraint
    L1L2ProjectionRegularizationParametrized
    - L1 regularization plus L2 (non-symmetric, parametrized) projection constraint
    - L1 regularization plus L2 (non-symmetric and symmetric, parametrized) projection constraint
"""

from keras import ops
from keras import regularizers


class L1L2ProjectionRegularization(regularizers.Regularizer):
    """
    L1 regularization plus L2 projection constraint.
    Given an input matrix V: N x N_coarse, the L1L2ProjectionRegularization term is
        ||V||_1 + ||V * V^T - I||_2
    """

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, V):
        return self.strength * ops.norm(V, ord=1) + self.strength * ops.norm(
            ops.matmul(V, ops.transpose(V)) - ops.eye(V.shape[0]), ord=2
        )

    def get_config(self):
        return {"strength": self.strength}


class L1L2ProjectionRegularizationParametrized(regularizers.Regularizer):
    """
    L1 regularization plus L2 projection constraint.
    Given an input matrix W: N_coarse x N, and a matrix V: N x N_coarse,
    the L1L2ProjectionRegularization term is
        ||W||_1 + ||W * V - I||_2
    """

    def __init__(self, strength: float, weight_matrix):
        self.strength = strength
        self.weight_matrix = weight_matrix

    def __call__(self, W):
        return self.strength * ops.norm(W, ord=1) + self.strength * ops.norm(
            ops.matmul(W, self.weight_matrix) - ops.eye(W.shape[0]), ord=2
        )

    def get_config(self):
        return {"strength": self.strength, "weight_matrix": self.weight_matrix}


class L1L2ProjectionRegularizationParametrizedSymmetric(regularizers.Regularizer):
    """
    L1 regularization plus L2 projection constraint.
    Given an input matrix W: N_coarse x N, and a matrix V: N x N_coarse,
    the L1L2ProjectionRegularizationParametrizedSymmetric term is
        ||W||_1 + ||W * W^T - I||_2 + ||W * V - I||_2
    """

    def __init__(self, strength: float, weight_matrix):
        """
        Initialize the regularizer.

        Args:
            strength (float): The regularization strength.
            weight_matrix (tf.Tensor): The matrix V.
        """
        self.strength = strength
        self.weight_matrix = weight_matrix

    def __call__(self, W):
        return (
            self.strength * ops.norm(W, ord=1)
            + self.strength
            * ops.norm(ops.matmul(W, ops.transpose(W)) - ops.eye(W.shape[0]), ord=2)
            + self.strength
            * ops.norm(ops.matmul(W, self.weight_matrix) - ops.eye(W.shape[0]), ord=2)
        )

    def get_config(self):
        return {"strength": self.strength, "weight_matrix": self.weight_matrix}


# WIP
# class L0Regularization(regularizers.Regularizer):
#     """
#     L0 regularization for matrices.
#     From https://arxiv.org/pdf/1712.01312.pdf
#     """
