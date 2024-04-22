"""
Module that generates data for the PseudoVcycle model.
"""

from abc import ABC, abstractmethod
from deprecated import deprecated
import ngsolve as ng
import numpy as np
import scipy.sparse as sp
import fes
import geo2d


class DataGenerator(ABC):
    """
    Abstract class for data generators.
    """

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the data generator.
        """

    @abstractmethod
    def from_smooth(self, num_samples: int):
        """
        Generate data from smooth functions.
        """

    @abstractmethod
    def from_random(self, num_samples: int):
        """
        Generate data from random vectors.
        """


class LaplaceDGen(DataGenerator):
    """
    Data generator for the Laplace equation.
    """

    def __init__(self, mesh: ng.Mesh, precond: str = "local", is_complex: bool = True):

        matrix_coeff = ng.CF((1.0, 0.0, 0.0, 1.0), dims=(2, 2))
        vector_coeff = ng.CF((0.0, 0.0))
        scalar_coeff = ng.CF(0.0)

        a, _, space = fes.convection_diffusion(
            mesh,
            matrix_coeff,
            vector_coeff,
            scalar_coeff,
            order=1,
            is_complex=is_complex,
        )
        c = ng.Preconditioner(a, precond)

        fes.assemble(a)

        self.laplace = a
        self.precond = c
        self.space = space
        self.dim = mesh.dim

    def __str__(self) -> str:
        description = (
            "Laplace Data Generator:\n"
            + "Generates data for the Laplace equation. It is a simple\n"
            + "second-order elliptic partial differential equation.\n"
        )
        return description

    def from_smooth(self, num_samples: int):
        gf = ng.GridFunction(self.space)
        x_data = ng.MultiVector(gf.vec, num_samples)
        for i in range(num_samples):
            if self.dim == 1:
                gf.Set(ng.sin(np.pi / 2 * i * ng.x))
            elif self.dim == 2:
                gf.Set(ng.sin(np.pi / 2 * i * ng.x) * ng.sin(np.pi / 2 * i * ng.y))
            elif self.dim == 3:
                gf.Set(
                    ng.sin(np.pi / 2 * i * ng.x)
                    * ng.sin(np.pi / 2 * i * ng.y)
                    * ng.sin(np.pi / 2 * i * ng.z)
                )
            else:
                raise ValueError("Not implemented for dimensions > 3.")
            x_data[i] = gf.vec
            x_data[i] += self.precond * (self.laplace.mat * gf.vec)
        return multivec_to_numpy(x_data)

    def from_random(self, num_samples: int):
        lap_mat = self.laplace.mat
        a_row, a_col, a_val = lap_mat.COO()
        lap_sp = sp.coo_matrix(
            (a_val, (a_row, a_col)), shape=(lap_mat.height, lap_mat.width)
        )
        gf = ng.GridFunction(self.space)
        x_data = ng.MultiVector(gf.vec, num_samples)
        for i in range(num_samples):
            x_data[i].SetRandom()
            x_data[i] += self.precond * (lap_sp @ x_data[i])
        return multivec_to_numpy(x_data)


def multivec_to_numpy(mv):
    """
    Convert a ngsolve.MultiVector to a numpy array.
    """
    return np.array([mv[i].FV().NumPy() for i in range(len(mv))])


def complex_to_real(x) -> np.ndarray:
    """
    Convert a complex numpy array to a real numpy array.
    """
    return np.concatenate((x.real, x.imag), axis=0)


def real_to_complex(x) -> np.ndarray:
    """
    Convert a real numpy array to a complex numpy array.
    """
    return x[: len(x) // 2] + 1j * x[len(x) // 2 :]


@deprecated(
    reason="Moving to a Class-based approach for data generation.",
    version="0.1.0",
    action="always",
)
def gd_convection_diffusion(
    num_samples: int, precond: str = "local", is_complex: bool = True
):
    """
    Generate data for the PseudoVcycle model, i.e., the encoder-decoder model.
    """
    mesh = ng.Mesh(geo2d.make_unit_square().GenerateMesh(maxh=0.1))
    matrix_coeff = ng.CF((1.0, 0.0, 0.0, 1.0), dims=(2, 2))
    vector_coeff = ng.CF((0.0, 0.0))
    scalar_coeff = ng.CF(0.0)

    a, _, space = fes.convection_diffusion(
        mesh, matrix_coeff, vector_coeff, scalar_coeff, order=1, is_complex=is_complex
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

    return multivec_to_numpy(x_data)


if __name__ == "__main__":
    xd = gd_convection_diffusion(10)
    for j in range(10):
        print(f"\nSample {j}:")
        print(xd[j])

    test_array = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    assert np.allclose(test_array, real_to_complex(complex_to_real(test_array)))

    laplace_dgen = LaplaceDGen(ng.Mesh(geo2d.make_unit_square().GenerateMesh(maxh=0.1)))

    print(laplace_dgen)
    print(laplace_dgen.from_random(10))
    print(laplace_dgen.from_smooth(10))

    print("All tests passed.")
