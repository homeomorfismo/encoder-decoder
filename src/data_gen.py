"""
Module that generates data for the PseudoVcycle model.
"""

from abc import ABC, abstractmethod  # from deprecated import deprecated
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
        description += f"\tSpace: {self.space}\n"
        description += f"\tPreconditioner: {self.precond}\n"
        # description += f"Laplace Matrix: {self.laplace}\n"
        return description

    def from_smooth(self, num_samples: int):
        gf = ng.GridFunction(self.space)
        x_data = ng.MultiVector(gf.vec, num_samples)
        for i in range(1, num_samples + 1):
            if self.dim == 1:
                gf.Set(
                    np.random.rand() * ng.sin(np.pi * i * ng.x)
                    + np.random.rand() * ng.cos(np.pi * i * ng.x)
                    + np.random.rand()
                )
            elif self.dim == 2:
                gf.Set(
                    np.random.rand()
                    * ng.sin(np.pi * i * ng.x)
                    * ng.sin(np.pi * i * ng.y)
                    + np.random.rand()
                )
            elif self.dim == 3:
                gf.Set(
                    np.random.rand()
                    * ng.sin(np.pi * i * ng.x)
                    * ng.sin(np.pi * i * ng.y)
                    * ng.sin(np.pi * i * ng.z)
                    + np.random.rand()
                )
            else:
                raise ValueError("Not implemented for dimensions > 3.")
            x_data[i - 1].data = gf.vec.data
            x_data[i - 1] -= self.precond * (self.laplace.mat * gf.vec)
            x_data[i - 1] /= np.linalg.norm(x_data[i - 1])
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
            x_data[i] /= np.linalg.norm(x_data[i])
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


def test_conversions():
    """
    Test the conversions between complex and real numpy arrays.
    """
    test_array = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    assert np.allclose(test_array, real_to_complex(complex_to_real(test_array)))


def test_data_gen():
    """
    Test the data generator.
    """
    laplace_dgen = LaplaceDGen(ng.Mesh(geo2d.make_unit_square().GenerateMesh(maxh=0.1)))
    print(laplace_dgen)

    gf = ng.GridFunction(laplace_dgen.space)
    gf.Set(ng.sin(np.pi * ng.x) * ng.sin(np.pi * ng.y))
    ng.Draw(gf, laplace_dgen.space.mesh, "sin(pi/2 * x)")
    input("Test Data Generator: Press Enter to continue...")

    x_data = laplace_dgen.from_smooth(1)
    gf_data = ng.GridFunction(laplace_dgen.space)
    gf_data.vec.FV().NumPy()[:] = x_data[0]
    ng.Draw(gf_data, laplace_dgen.space.mesh, "smooth data")
    input("Test Data Generator: Press Enter to continue...")


def test_data_gen_sines(num_samples: int = 5):
    """
    Draw several sine functions.
    """
    laplace_dgen = LaplaceDGen(ng.Mesh(geo2d.make_unit_square().GenerateMesh(maxh=0.1)))
    x_data = laplace_dgen.from_smooth(num_samples)
    gf_data = ng.GridFunction(laplace_dgen.space)
    for i in range(num_samples):
        gf_data.vec.FV().NumPy()[:] = x_data[i]
        ng.Draw(gf_data, laplace_dgen.space.mesh, f"smooth data {i}")
        input("Test Data Generator: Press Enter to continue...")


if __name__ == "__main__":
    test_conversions()
    test_data_gen()
    test_data_gen_sines()
