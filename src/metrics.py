"""
This file contains the custom metrics that are used in the model.
"""

from keras.metrics import Metric
from keras import ops
from keras import regularizers


class SymL1Regularization(regularizers.Regularizer):
    """
    L1 regularization for symmetric matrices.
    Given an input matrix V, the symmetric L1 regularization term is the sum of
        ||W - V^T||_1 + ||W||_1
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
        """
        Compute the regularization term.

        Args:
            W (tf.Tensor): The matrix W.

        Returns:
            tf.Tensor: The regularization term.
        """
        return self.strength * ops.norm(
            ops.transpose(W) - self.weight_matrix, ord=1
        ) + self.strength * ops.norm(W, ord=1)

    def get_config(self):
        """
        Get the configuration of the regularizer.

        Returns:
            dict: The configuration of the regularizer.
        """
        return {"strength": self.strength, "transpose": self.transpose}


class IdL1Regularization(regularizers.Regularizer):
    """
    L1 regularization for identity matrices.
    Given an input matrix V, the identity L1 regularization term is
        ||W*V - I||_1
    """

    def __init__(self, strength: float, weight_matrix):
        """
        Initialize the regularizer.

        Args:
            strength (float): The regularization strength.
            weight_matrix (tf.Tensor): The matrix V.
        """
        self.strength = strength
        self.identity = ops.eye(weight_matrix.shape[0])
        self.weight_matrix = ops.transpose(weight_matrix)

    def __call__(self, W):
        """
        Compute the regularization term.

        Args:
            W (tf.Tensor): The matrix W.

        Returns:
            tf.Tensor: The regularization term.
        """
        return self.strength * ops.norm(
            ops.matmul(ops.transpose(W), self.weight_matrix) - self.identity, ord=1
        ) + self.strength * ops.norm(W, ord=1)

    def get_config(self):
        """
        Get the configuration of the regularizer.

        Returns:
            dict: The configuration of the regularizer.
        """
        return {"strength": self.strength, "identity": self.identity}


class SymIdL1Regularization(regularizers.Regularizer):
    """
    L1 regularization for symmetric identity matrices.
    Given an input matrix V, the symmetric identity L1 regularization term is the sum of
        ||W*V - I||_1 + ||W - V^T||_1 + ||W||_1
    """

    def __init__(self, strength: float, weight_matrix):
        """
        Initialize the regularizer.

        Args:
            strength (float): The regularization strength.
            weight_matrix (tf.Tensor): The matrix V.
        """
        self.strength = strength
        self.identity = ops.eye(weight_matrix.shape[0])
        self.weight_matrix = ops.transpose(weight_matrix)

    def __call__(self, W):
        """
        Compute the regularization term.

        Args:
            W (tf.Tensor): The matrix W.

        Returns:
            tf.Tensor: The regularization term.
        """
        return (
            self.strength
            * ops.norm(
                ops.matmul(ops.transpose(W), self.weight_matrix) - self.identity, ord=1
            )
            + self.strength * ops.norm(W - self.weight_matrix, ord=1)
            + self.strength * ops.norm(W, ord=1)
        )

    def get_config(self):
        """
        Get the configuration of the regularizer.

        Returns:
            dict: The configuration of the regularizer.
        """
        return {
            "strength": self.strength,
            "identity": self.identity,
            "transpose": self.transpose,
        }
