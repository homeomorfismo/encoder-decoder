"""
Module that generates data for the PseudoVcycle model.
"""

import numpy as np
import tensorflow as tf
import ngsolve as ng
import scipy.sparse as sp
import fes
import geo2d


def generate_data(num_samples: int, precond: str = "local"):
    """
    Generate data for the PseudoVcycle model, i.e., the encoder-decoder model.
    """
    mesh = ng.Mesh(geo2d.make_unit_square().GenerateMesh(maxh=0.1))
    matrix_coeff = ng.CF((1.0, 0.0, 0.0, 1.0), dims=(2, 2))
    vector_coeff = ng.CF((0.0, 0.0))
    scalar_coeff = ng.CF(0.0)

    a, _, space = fes.convection_diffusion(
        mesh, matrix_coeff, vector_coeff, scalar_coeff
    )
    c = ng.Preconditioner(a, precond)
    fes.assemble(a)

    a_row, a_col, a_val = a.mat.COO()
    a_sp = sp.coo_matrix((a_val, (a_row, a_col)), shape=(a.mat.height, a.mat.width))
    # c_row, c_col, c_val = c.mat.COO()
    # c_sp = sp.coo_matrix((c_val, (c_row, c_col)), shape=(c.mat.height, c.mat.width))

    gf = ng.GridFunction(space)
    x_data = ng.MultiVector(gf.vec, num_samples)
    for i in range(num_samples):
        x_data[i].SetRandom()
        x_data[i] += c * (a_sp @ x_data[i])  # Cursed line

    return x_data


if __name__ == "__main__":
    x = generate_data(10)
    for i in range(10):
        print(f"Sample {i}:")
        print(x[i])
