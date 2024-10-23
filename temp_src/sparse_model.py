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


class SparseVcycle(nn.Module):
    """
    Sparse Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
    All tensors are assumed to be sparse.
    """

    def __init__(self, **kwargs):
        """
        Constructor for the SparseVcycle model.

        Required parameters:
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
