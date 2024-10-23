"""
Module that constains internal functions for the encoder-decoder model.

We have the following functions:
- get_device: Get the device to use for the model.
"""

import torch
import numpy as np
import networkx as nx
from networkx.algorithms import approximation as approx


def get_device():
    """
    Get the device (GPU, MPS, or CPU).
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    return device


def get_near_nullspace_from_matrix(matrix, smoother, tol=1e-7):
    """
    Get a near nullspapce of a matrix.

    Inputs:
    - matrix (scipy.sparse.csr_matrix): Sparse matrix. It is assumed to be
    square.
    - smoother (scipy.sparse.csr_matrix): Smoother matrix
    - tol (float): Relative tolerance for the near nullspace.

    Outputs:
    - near_kernel (numpy.ndarray): Near nullspace of the matrix.
    """
    max_iter = 100
    rel_tol = 0
    current_iter = 0
    x_k1 = np.random.rand(matrix.shape[1])
    b_k1 = matrix @ x_k1
    while rel_tol < tol and current_iter < max_iter:
        current_iter += 1
        x_k = x_k1
        b_k = b_k1
        b_k = matrix @ x_k
        r_k = smoother @ b_k
        x_k1 = x_k - r_k
        b_k1 = matrix @ x_k1
        rel_tol = np.dot(b_k1, x_k1) / np.dot(b_k, x_k)
    near_kernel = x_k1 / np.sqrt(np.dot(x_k1, b_k1))
    return near_kernel


def get_near_nullspace_from_function(matrix, smoother, tol=1e-7):
    """
    Get a near nullspapce of a matrix. Assumes the smoother is a function
    that takes a matrix and a vector and returns a vector.

    Inputs:
    - matrix (scipy.sparse.csr_matrix): Sparse matrix. It is assumed to be
    square.
    - smoother (function): Smoother function.
    - tol (float): Relative tolerance for the near nullspace.

    Outputs:
    - near_kernel (numpy.ndarray): Near nullspace of the matrix.
    """
    max_iter = 100
    rel_tol = 0
    current_iter = 0
    x_k1 = np.random.rand(matrix.shape[1])
    b_k1 = matrix @ x_k1
    while rel_tol < tol and current_iter < max_iter:
        current_iter += 1
        x_k = x_k1
        b_k = b_k1
        b_k = matrix @ x_k
        r_k = smoother(matrix, b_k)
        x_k1 = x_k - r_k
        b_k1 = matrix @ x_k1
        rel_tol = np.dot(b_k1, x_k1) / np.dot(b_k, x_k)
    near_kernel = x_k1 / np.sqrt(np.dot(x_k1, b_k1))
    return near_kernel


def get_fine_to_coarse_table(matrix, vector, theta=0.5):
    """
    Produces a matrix/table that maps fine dofs to coarse dofs.
    Upon input of a matrix, we construct the sparsity graph, and then
    we find the maximum independent set of the graph. These nodes are the
    indices of the coarse dofs.

    Inputs:
    - matrix (scipy.sparse.csr_matrix): Sparse matrix.
    - vector (numpy.ndarray): Near nullspace of the matrix.
    - theta (float): Threshold for the fine-to-coarse mapping.

    Outputs:
    - projector (scipy.sparse.csr_matrix): Sparse matrix that maps fine
    dofs to coarse dofs.
    """
    # Aggregation technique:
    # 1 Per i row, get nonzero entries of the matrix. These are the neighbors
    # of the ith node. (We may use the graph)
    # 2 Divide the (ith) neighbors into two sets:
    # a_ii + sum_{j in N1} a_ij * w_j / w_i
    # approx - sum_{j in N2} a_ij * w_j / w_i
    # where the sum on the right is negative (i.e., RHS > 0).
    # 3 Ensure that N2 has a coarse node. If not, add the coarse node to N2.
    # N2 can be constructed from N2 = N - {i}, where N is the set of neighbors,
    # and then reduce it by removing some nodes.
    # 4 We define P(i, j_c) to be non-zero if:
    # a_i,j_c * w_j_c / w_i >= theta * max_{j in N2 cap N_c} a_i,j * w_j / w_i
    graph = nx.from_scipy_sparse_matrix(matrix)
    fine_nodes = list(graph.nodes)
    coarse_nodes = approx.maximum_independent_set(graph)
    fine_to_coarse = {}
    for i in fine_nodes:
        neighbors = list(graph.neighbors(i))
        N2 = set(neighbors) - {i}
        N1 = set(neighbors) - N2
    raise NotImplementedError
