"""
Module that contains the encoder-decoder model for mimicking a
V-Cycle in a Multigrid solver. We focus on sparse and semi-sparse
models.

We have four models:
- DenseVcycle: Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
- DenseMG: Encoder-Decoder model for mimicking a Multigrid solver.
- SparseVcycle: Sparse Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
- SparseMG: Sparse Encoder-Decoder model for mimicking a Multigrid solver.

We base these models on unpublished work by Dr. Panayot Vassilevski and old code
written by Gabriel Pinochet-Soto.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as functional
import internal


class SparseVcycle(nn.Module):
    """
    Sparse Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
    All tensors are assumed to be sparse.
    """

    def __init__(self, input_shape=None, coarse_shape=None, bool_matrix=None, **kwargs):
        """
        Constructor for the SparseVcycle model.

        Required parameters:
        - input_shape (tuple): Shape of the input tensor.
        - coarse_shape (tuple): Shape of the coarse (smallest) tensor.

        Other parameters are stored in a dictionary.
        """
        assert input_shape is not None, "Input shape must be provided."
        assert coarse_shape is not None, "Coarse shape must be provided."
        assert bool_matrix is not None, "Matrix must be provided."
        shape_matrix = bool_matrix.shape
        # TODO: Projector is tall
        # assert shape_matrix[0] = coarse_shape[0], "Matrix must have the same number of rows as the coarse shape."
        # assert shape_matrix[1] = input_shape[0], "Matrix must have the same number of columns as the input shape."
        super().__init__()

        self._name = "SparseVcycle"
        self._input_shape = input_shape
        self._coarse_shape = coarse_shape
        self._sparsity_projector = bool_matrix

        self._dict = kwargs

        self.encoder = nn.Linear(
            in_features=self._input_shape[-1],
            out_features=self._coarse_shape[-1],
            bias=False,
        )

        self.decoder = nn.Linear(
            in_features=self._coarse_shape[-1],
            out_features=self._input_shape[-1],
            bias=False,
        )

        # Only trainable weights are given by the sparsity projector
        raise NotImplementedError

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError


class SparseMG(nn.Module):
    """
    Sparse Encoder-Decoder model for mimicking a Multigrid solver.
    All tensors are assumed to be sparse.
    """

    def __init__(self, matrix=None, **kwargs):
        """
        Constructor for the SparseMG model.

        Required parameters:
        - matrix (torch.Tensor): Matrix tensor.
        - input_shape (tuple): Shape of the input tensor.
        - coarse_shape (tuple): Shape of the coarse (smallest) tensor.

        Other parameters are stored in a dictionary.
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError
