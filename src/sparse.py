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

# import pyamg.aggregation.aggregate as agg
import pyamg.aggregation as agg
import numpy as np
from typing import Any

# Parameters
__DIM__ = 100
__NUM_CLUSTERS__ = 3


class AggregationType(Enum):
    standard_aggregation = "standard_aggregation"
    naive_aggregation = "naive_aggregation"
    # pairwise_aggregation = "pairwise_aggregation"
    lloyd_aggregation = "lloyd_aggregation"
    balanced_lloyd_aggregation = "balanced_lloyd_aggregation"


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


if __name__ == "__main__":
    matrix = sp.csr_matrix(np.random.rand(__DIM__, __DIM__))
    print(
        f"\nMatrix shape: {matrix.shape}"
        f"\nMatrix sparsity pattern:\n{matrix.toarray()}"
    )

    for agg_type in AggregationType:
        if agg_type == AggregationType.balanced_lloyd_aggregation:
            projector = pyamg_get_sparsity_pattern_projector(
                matrix, agg_type, num_clusters=__NUM_CLUSTERS__
            )
        else:
            projector = pyamg_get_sparsity_pattern_projector(matrix, agg_type)
        print(
            f"\n\t-> Aggregation type: {agg_type.name}"
            f"\n\t-> Shape: {projector[0].shape}"
            f"\n\t-> Sparsity pattern projector:\n{projector[0].toarray()}"
        )
