"""
Module defining aggregation/coarsening/interpolation routines for sparse matrices using PyAMG.
These matrices define sparsity patterns for a (semi)-sparse encoder-decoder method, with JAX.
"""

from enum import Enum
import scipy.sparse as sp
import pyamg.aggregation as agg


class AggregationType(Enum):
    STANDARD = "standard_aggregation"
    NAIVE = "naive_aggregation"
    PAIRWISE = "pairwise_aggregation"
    LLOYD = "lloyd_aggregation"
    BALANCED_LLOYD = "balanced_lloyd_aggregation"


def pyamg_get_sparsity_pattern_projector(
    matrix: sp.csr_matrix, agg_type: str
) -> sp.csr_matrix:
    """
    Get the sparsity pattern projector using PyAMG.

    Args:
    - matrix (scipy.sparse.csr_matrix): Sparse matrix.
    - agg_type (str): Aggregation type.

    Returns:
    - scipy.sparse.csr_matrix: Sparse matrix that maps fine dofs to coarse dofs.
    """
    try:
        aggregation_method = getattr(agg, AggregationType[agg_type].value)
    except KeyError:
        raise ValueError(f"Invalid aggregation type: {agg_type}")
    return aggregation_method(matrix)


if __name__ == "__main__":
    import numpy as np

    # Test sparsity pattern projector
    matrix = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    projector = pyamg_get_sparsity_pattern_projector(matrix, "STANDARD")
    print(projector.toarray())
    projector = pyamg_get_sparsity_pattern_projector(matrix, "NAIVE")
    print(projector.toarray())
    projector = pyamg_get_sparsity_pattern_projector(matrix, "PAIRWISE")
    print(projector.toarray())
    projector = pyamg_get_sparsity_pattern_projector(matrix, "LLOYD")
    print(projector.toarray())
    projector = pyamg_get_sparsity_pattern_projector(matrix, "BALANCED_LLOYD")
    print(projector.toarray())
