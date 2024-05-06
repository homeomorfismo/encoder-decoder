"""
This file contains custom regularizers for symmetric and identity matrices.
"""

from keras import ops
from keras import regularizers

# from keras import activations


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
        return {"strength": self.strength, "weight_matrix": self.weight_matrix}


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
            "weight_matrix": self.weight_matrix,
        }


# WIP
# class L0Regularization(regularizers.Regularizer):
#     """
#     L0 regularization for matrices.
#     From https://arxiv.org/pdf/1712.01312.pdf
#     """

#     def __init__(self, interval_strech: tuple = (-.1, 1.1), strength: float = 1e-3):
#         """
#         Initialize the regularizer.

#         Args:
#             strength (float): The regularization strength.
#         """
#         assert interval_strech[0] < 0 and interval_strech[1] > 1.0, "Interval must contain (0, 1)"

#         self.strength = strength
#         self.start = interval_strech[0]
#         self.end = interval_strech[1]

#     def __call__(self, W):
#         """
#         Compute the regularization term.

#         Args:
#             W (tf.Tensor): The matrix W.

#         Returns:
#             tf.Tensor: The regularization term.
#         """
#         temp = activations.sigmoid(j
#         return self.strength * ops.count_nonzero(W)

#     def get_config(self):
#         """
#         Get the configuration of the regularizer.

#         Returns:
#             dict: The configuration of the regularizer.
#         """
#         return {"strength": self.strength}
