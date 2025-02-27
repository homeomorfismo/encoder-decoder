"""
Module defining aggregation/coarsening/interpolation routines for sparse matrices using PyAMG.
These matrices define sparsity patterns for a (semi)-sparse encoder-decoder method, with JAX.

From PyAMG documentation:
- pairwise_aggregation(A, matchings=2, theta=0.25, norm='min', compute_P=False)
- lloyd_aggregation(C, ratio=0.03, distance='unit', maxiter=10)
- balanced_lloyd_aggregation(C, num_clusters=None)
"""

from enum import Enum
import scipy.sparse as sp
import pyamg.aggregation as agg
import pyamg.strength as strength
import numpy as np
from typing import Any


class AggregationType(Enum):
    STANDARD = "standard_aggregation"
    NAIVE = "naive_aggregation"
    # PAIRWISE = "pairwise_aggregation"
    LLOYD = "lloyd_aggregation"
    BALANCED_LLOYD = "balanced_lloyd_aggregation"


class StrengthOfConnectionType(Enum):
    SYMMETRIC = "symmetric"
    CLASSICAL = "classical"
    # DISTANCE = "distance"
    ENERGY_BASED = "energy_based"
    ALGEBRAIC = "algebraic"
    AFFINITY = "affinity"


def strength_of_connection(
    matrix: sp.csr_matrix,
    strength_type: StrengthOfConnectionType,
    **kwargs: Any,
) -> sp.csr_matrix:
    """
    Get the strength of connection matrix using PyAMG.

    Args:
    - matrix (scipy.sparse.csr_matrix): Sparse matrix.
    - strength_type (StrengthOfConnectionType): Strength of connection type.

    Returns:
    - scipy.sparse.csr_matrix: Sparse matrix that defines the strength of connection.
    """
    try:
        if strength_type in [
            StrengthOfConnectionType.ALGEBRAIC,
            StrengthOfConnectionType.AFFINITY,
        ]:
            strength_method = getattr(
                strength, strength_type.value + "_distance"
            )
        else:
            strength_method = getattr(
                strength,
                strength_type.value + "_strength_of_connection",
            )
    except AttributeError:
        raise ValueError(
            f"Invalid strength of connection type: {strength_type}"
        )
    return strength_method(matrix, **kwargs)


def aggregation_matrix(
    matrix: sp.csr_matrix,
    agg_type: AggregationType,
    **kwargs: Any,
) -> sp.csr_matrix:
    """
    Get the aggregation matrix using PyAMG.

    Args:
    - matrix (scipy.sparse.csr_matrix): Sparse matrix.
    - agg_type (AggregationType): Aggregation type.

    Returns:
    - scipy.sparse.csr_matrix: Sparse matrix that maps fine dofs to coarse dofs.
    """
    try:
        aggregation_method = getattr(agg, agg_type.value)
    except AttributeError:
        raise ValueError(f"Invalid aggregation type: {agg_type}")
    return aggregation_method(matrix, **kwargs)
