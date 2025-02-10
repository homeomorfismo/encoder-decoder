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
import numpy as np
from typing import Any

# Parameters
__DIM__ = 100
__NUM_CLUSTERS__ = 3


class AggregationType(Enum):
    STANDARD = "standard_aggregation"
    NAIVE = "naive_aggregation"
    # PAIRWISE = "pairwise_aggregation"
    LLOYD = "lloyd_aggregation"
    BALANCED_LLOYD = "balanced_lloyd_aggregation"


def pyamg_get_sparsity_pattern_projector(
    matrix: sp.csr_matrix,
    agg_type: AggregationType,
    **kwargs: Any,
) -> sp.csr_matrix:
    """
    Get the sparsity pattern projector using PyAMG.

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
