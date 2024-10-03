"""
Losses for training the model.
"""

from keras import ops


def projected_l2_loss(matrix, y_true, y_pred):
    """
    Mean squared error loss function that takes into account the fact that the
    output is a projection of the true value onto a subspace.
    Requires to use partial functions to pass the matrix as an argument.
    """
    return ops.norm(ops.matmul(y_true - y_pred, matrix), ord=2)


def l2_l1_loss(y_true, y_pred):
    """
    Loss function that combines L2 and L1 norms.
    """
    return ops.norm(y_true - y_pred, ord=2) + ops.norm(y_true - y_pred, ord=1)
