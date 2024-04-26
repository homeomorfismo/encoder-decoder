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
    def make_operator(self):
        """
        Make a linear operator.
        """

    @abstractmethod
    def make_sparse_operator(self):
        """
        Make a sparse linear operator.
        """

    @abstractmethod
    def make_solver(self):
        """
        Make a solver.
        """

    @abstractmethod
    def make_sparse_solver(self):
        """
        Make a sparse solver.
        """

    @abstractmethod
    def make_data(self, x):
        """
        Make data.
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

    def __init__(
        self,
        mesh: ng.Mesh,
        tol: float = 1e-5,
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

        self.ngsolve_operator = a
        self.operator = self.make_operator()
        self.sparse_operator = self.make_sparse_operator()
        self.tol = tol

        self.solver = self.make_solver()
        self.sparse_solver = self.make_sparse_solver()

    def make_operator(self):
        """
        Make a linear operator.

        Applys the NGSolve operator to a numpy array.
        Defines a scipy linear operator.
        """

        def operator(x):
            self.__ng_input.vec.FV().NumPy()[:] = x
            self.__ng_output.vec.data = self.ngsolve_operator.mat * self.__ng_input.vec
            return self.__ng_output.vec.FV().NumPy()

        scipy_operator = sp.linalg.LinearOperator(
            (self.ngsolve_operator.mat.height, self.ngsolve_operator.mat.width),
            matvec=operator,
        )
        return scipy_operator

    def make_sparse_operator(self):
        """
        Make a sparse linear operator.

        Returns a scipy sparse matrix in CSR format, from the NGSolve operator.
        """
        a_row, a_col, a_val = self.ngsolve_operator.mat.COO()
        return sp.csr_matrix(
            (a_val, (a_row, a_col)),
            shape=(self.ngsolve_operator.mat.height, self.ngsolve_operator.mat.width),
        )

    def make_solver(self):
        """
        Make a solver.
        """

        def error(*args, **kwargs):
            raise NotImplementedError("Direct solver not implemented.")

        return error

    def make_sparse_solver(self):
        """
        Make a sparse solver.

        Returns a sparse solver based on the ILU preconditioner,
        for the sparse operator (the CSR matrix).
        """
        return sp.linalg.spilu(self.sparse_operator, drop_tol=self.tol).solve

    def make_data(self, x):
        """
        Make data.

        Returns the residual of the NGSolve operator applied to x, normalized.
        """
        z = x.copy()
        ax = self.sparse_operator @ z
        z -= self.sparse_solver(ax)
        # z /= np.linalg.norm(z)
        z /= np.max(np.abs(z))
        return z

    def from_smooth(self, num_samples: int):
        """
        Generate data from smooth functions.

        Returns a numpy array of num_samples samples.
        """
        gf = ng.GridFunction(self.__space)
        x_data = ng.MultiVector(gf.vec, num_samples)
        for i in range(1, num_samples + 1):
            if self.__dim == 1:
                gf.Set(
                    np.random.rand() * ng.sin(np.pi * i * ng.x)
                    + np.random.rand() * ng.cos(np.pi * i * ng.x)
                    + np.random.rand()
                )
            elif self.__dim == 2:
                gf.Set(
                    np.random.rand()
                    * ng.sin(np.pi * i * ng.x)
                    * ng.sin(np.pi * i * ng.y)
                    + np.random.rand()
                )
            elif self.__dim == 3:
                gf.Set(
                    np.random.rand()
                    * ng.sin(np.pi * i * ng.x)
                    * ng.sin(np.pi * i * ng.y)
                    * ng.sin(np.pi * i * ng.z)
                    + np.random.rand()
                )
            else:
                raise ValueError("Not implemented for dimensions > 3.")
            x_data[i - 1].FV().NumPy()[:] = self.make_data(gf.vec.FV().NumPy())
        return multivec_to_numpy(x_data)

    def from_random(self, num_samples: int):
        """
        Generate data from random vectors.

        Returns a numpy array of num_samples samples.
        """
        gf = ng.GridFunction(self.__space)
        x_data = ng.MultiVector(gf.vec, num_samples)
        for i in range(num_samples):
            x_data[i].SetRandom()
            x_data[i].FV().NumPy()[:] = self.make_data(x_data[i].FV().NumPy())
        return multivec_to_numpy(x_data)


####################################################################################################
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


####################################################################################################
def test_conversions():
    """
    Test the conversions between complex and real numpy arrays.
    """
    test_array = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    assert np.allclose(test_array, real_to_complex(complex_to_real(test_array)))


def test_data_gen_vs_direct():
    """
    Test the data generator.
    """
    mesh = ng.Mesh(geo2d.make_unit_square().GenerateMesh(maxh=0.1))
    generate = LaplaceDGen(mesh, tol=1e-2, is_complex=True)

    gf = ng.GridFunction(generate.ngsolve_operator.space)
    gf.Set(ng.sin(np.pi * ng.x) * ng.sin(np.pi * ng.y))
    ng.Draw(gf, generate.ngsolve_operator.space.mesh, "sin(pi*x)*sin(pi*y)")
    input("sin(pi*x)*sin(pi*y): Press Enter to continue...")

    x_data = generate.from_smooth(1)
    gf_data = ng.GridFunction(generate.ngsolve_operator.space)
    gf_data.vec.FV().NumPy()[:] = x_data[0]
    ng.Draw(gf_data, mesh, "smooth data")
    input("Normalized residual: Press Enter to continue...")


def test_data_gen_sines(num_samples: int = 5):
    """
    Draw several sine functions.
    """
    mesh = ng.Mesh(geo2d.make_unit_square().GenerateMesh(maxh=0.1))
    generate = LaplaceDGen(mesh, tol=1e-2, is_complex=True)
    x_data = generate.from_smooth(num_samples)
    gf_data = ng.GridFunction(generate.ngsolve_operator.space)
    for i in range(num_samples):
        gf_data.vec.FV().NumPy()[:] = x_data[i]
        ng.Draw(gf_data, mesh, f"Res Sine {i}")
        input("\tNormalized residual: Press Enter to continue...")


def test_data_gen_random(num_samples: int = 5):
    """
    Draw several random functions.
    """
    mesh = ng.Mesh(geo2d.make_unit_square().GenerateMesh(maxh=0.1))
    generate = LaplaceDGen(mesh, tol=1e-2, is_complex=True)
    x_data = generate.from_random(num_samples)
    gf_data = ng.GridFunction(generate.ngsolve_operator.space)
    for i in range(num_samples):
        gf_data.vec.FV().NumPy()[:] = x_data[i]
        ng.Draw(gf_data, mesh, f"Res Random {i}")
        input("\tNormalized residual: Press Enter to continue...")


if __name__ == "__main__":
    test_conversions()
    test_data_gen_vs_direct()
    test_data_gen_sines()
    test_data_gen_random()
