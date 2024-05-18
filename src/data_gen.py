"""
Module that generates data for the PseudoVcycle model.
"""

import ngsolve as ng
import numpy as np
import scipy.sparse as sp
import fes

from solver import symmetric_gauss_seidel
from geo2d import make_unit_square


class LaplaceDGen:
    """
    Data generator for the Laplace equation.
    """

    def __init__(
        self,
        mesh: ng.Mesh,
        tol: float = 1e-3,
        is_complex: bool = True,
    ):

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
        fes.assemble(a)

        self.__space = space
        self.__dim = space.mesh.dim
        self.__ng_input = ng.GridFunction(self.__space)
        self.__ng_output = ng.GridFunction(self.__space)

        a_row, a_col, a_val = a.mat.COO()
        self.operator = sp.csr_matrix(
            (a_val, (a_row, a_col)),
            shape=(a.mat.height, a.mat.width),
        ).todense()
        self.tol = tol

    def make_data(self, x):
        """
        Make data.

        Returns the residual of the NGSolve operator applied to x, normalized.
        """
        z = x.copy()
        ax = np.ravel(np.dot(self.operator, x))
        ma_x = np.ones_like(ax)
        z -= symmetric_gauss_seidel(self.operator, ma_x, ax, tol=self.tol, max_iter=100)
        z /= np.max(np.abs(z))
        return z

    def from_smooth(self, num_samples: int, field: str = "real"):
        """
        Generate data from smooth functions.

        Returns a numpy array of num_samples samples.
        """
        if field == "real":
            alpha = np.random.rand()
        elif field == "imag":
            alpha = np.random.rand() * 1.0j
        else:
            alpha = np.random.rand() + np.random.rand() * 1.0j
        gf = ng.GridFunction(self.__space)
        x_data = ng.MultiVector(gf.vec, num_samples)
        for i in range(1, num_samples + 1):
            if self.__dim == 1:
                a = np.random.rand()
                gf.Set(
                    (np.random.rand() * ng.sin(np.pi * a * i * ng.x) + np.random.rand())
                    * alpha
                )
            elif self.__dim == 2:
                a = np.random.rand()
                b = np.random.rand()
                gf.Set(
                    (
                        np.random.rand()
                        * ng.sin(np.pi * a * i * ng.x)
                        * ng.sin(np.pi * b * i * ng.y)
                        + np.random.rand()
                    )
                    * alpha
                )
            elif self.__dim == 3:
                a = np.random.rand()
                b = np.random.rand()
                c = np.random.rand()
                gf.Set(
                    (
                        np.random.rand()
                        * ng.sin(np.pi * a * i * ng.x)
                        * ng.sin(np.pi * b * i * ng.y)
                        * ng.sin(np.pi * c * i * ng.z)
                        + np.random.rand()
                    )
                    * alpha
                )
            else:
                raise ValueError("Not implemented for dimensions > 3.")
            x_data[i - 1].FV().NumPy()[:] = self.make_data(gf.vec.FV().NumPy())
        return multivec_to_numpy(x_data)

    def from_random(self, num_samples: int, field: str = "real"):
        """
        Generate data from random vectors.

        Returns a numpy array of num_samples samples.
        """
        if field == "real":
            alpha = np.random.rand()
        elif field == "imag":
            alpha = np.random.rand() * 1.0j
        else:
            alpha = np.random.rand() + np.random.rand() * 1.0j
        gf = ng.GridFunction(self.__space)
        x_data = ng.MultiVector(gf.vec, num_samples)
        for i in range(num_samples):
            x_data[i].SetRandom()
            x_data[i].FV().NumPy()[:] *= alpha
            x_data[i].FV().NumPy()[:] = self.make_data(x_data[i].FV().NumPy())
        return multivec_to_numpy(x_data)

    def get_gf(self, name="gf"):
        """
        Get a GridFunction.
        """
        return ng.GridFunction(self.__space, name=name)


####################################################################################################
def multivec_to_numpy(mv):
    """
    Convert a ngsolve.MultiVector to a numpy array.
    """
    return np.array([mv[i].FV().NumPy() for i in range(len(mv))])


def complex_to_real(x) -> np.ndarray:
    """
    Convert a complex numpy array to a real numpy array.
    [z1, z2, z3, ...] -> [ [z1.real, z2.real, ...], [z1.imag, z2.imag, ...] ]
    """
    return np.concatenate((x.real, x.imag), axis=0)


def real_to_complex(x) -> np.ndarray:
    """
    Unconvert a real numpy array to a complex numpy array.
    [ [z1.real, z2.real, ...], [z1.imag, z2.imag, ...] ] -> [z1, z2, z3, ...]
    """
    return x[: x.shape[0] // 2] + 1j * x[x.shape[0] // 2 :]


####################################################################################################
def test_conversions():
    """
    Test the conversions between complex and real numpy arrays.
    """
    test_array = np.array([[1 + 2j, 3 + 4j, 5 + 6j], [7 + 8j, 9 + 10j, 11 + 12j]])
    print(f"test_array:\n {test_array}\n" f"test_array.shape: {test_array.shape}\n")
    print(
        f"complex_to_real(test_array):\n {complex_to_real(test_array)}\n"
        f"complex_to_real(test_array).shape: {complex_to_real(test_array).shape}\n"
    )
    print(
        f"real_to_complex(complex_to_real(test_array)):\n {real_to_complex(complex_to_real(test_array))}\n"
        f"real_to_complex(complex_to_real(test_array)).shape: {real_to_complex(complex_to_real(test_array)).shape}\n"
    )
    assert np.allclose(test_array, real_to_complex(complex_to_real(test_array)))


def test_smooth_data():
    """
    Test the generation of data from smooth functions.
    """
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.05))
    dgen = LaplaceDGen(mesh, tol=1e-2, is_complex=False)
    x_data = dgen.from_smooth(10, field="real")
    gf = dgen.get_gf(name="smooth")
    for i in range(10):
        gf.vec.FV().NumPy()[:] = x_data[i]
        ng.Draw(gf, mesh, f"smooth_{i}")
    x_data = dgen.from_random(10, field="real")
    for i in range(10):
        gf.vec.FV().NumPy()[:] = x_data[i]
        ng.Draw(gf, mesh, f"random_{i}")


if __name__ == "__main__":
    test_conversions()
    test_smooth_data()
