"""
Module that generates data for the PseudoVcycle model.
"""

from abc import ABC, abstractmethod
from jax import random
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import ngsolve as ng

# local imports
import fes
from solver import coordinate_descent
from geo2d import make_unit_square

# Parameters
__MAXH__ = 0.1


class DataGenerator(ABC):
    """
    Abstract base class for data generators.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the data generator.
        """
        pass

    @abstractmethod
    def make_data(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Generate data based on input x.

        Args:
            x: jnp.ndarray
                Input data.

        Returns:
            jnp.ndarray
                Generated data.
        """
        pass

    @abstractmethod
    def generate_samples(self, num_samples: int) -> jnp.ndarray:
        """
        Generate a number of data samples.

        Args:
            num_samples: int
                Number of samples to generate.

        Returns:
            jnp.ndarray
                Array of generated samples.
        """
        pass

    @abstractmethod
    def get_gf(self, name: str = "gf") -> ng.GridFunction:
        """
        Get a GridFunction for the data generator.

        Args:
            name: str
                Name of the GridFunction.

        Returns:
            GridFunction
                GridFunction object.
        """
        pass


class HelmholtzDGen(DataGenerator):
    def __init__(
        self,
        mesh: ng.Mesh,
        tol: float = 1e-3,
        order: int = 1,
        iterations: int = 1_000,
        is_complex: bool = True,
        is_dirichlet: bool = True,
    ):
        matrix_coeff = ng.CF((1.0, 0.0, 0.0, 1.0), dims=(2, 2))
        vector_coeff = ng.CF((0.0, 0.0))
        scalar_coeff = ng.CF(1.0)
        a, _, space = fes.convection_diffusion(
            mesh,
            matrix_coeff,
            vector_coeff,
            scalar_coeff,
            order=order,
            is_complex=is_complex,
            is_dirichlet=is_dirichlet,
        )
        fes.assemble(a)

        self.space = space
        a_row, a_col, a_val = a.mat.COO()
        self.operator = jnp.array(
            sp.csr_matrix(
                (a_val, (a_row, a_col)), shape=(a.mat.height, a.mat.width)
            ).todense()
        )
        self.free_dofs = jnp.array(list(space.FreeDofs()))
        self.rest_operator = (
            self.operator.at[~self.free_dofs]
            .set(0)
            .at[:, ~self.free_dofs]
            .set(0)
        )
        self.rest_operator = self.rest_operator.at[
            jnp.diag_indices_from(self.rest_operator)
        ].set(self.operator.diagonal())

        self.tol = tol
        self.iterations = iterations
        self.is_complex = is_complex

    def make_data(self, x: jnp.ndarray) -> jnp.ndarray:
        z = x.copy()
        b = jnp.dot(self.operator, z)
        ma_x = jnp.ones_like(b)
        r = b - jnp.dot(self.operator, ma_x)

        for _ in range(self.iterations):
            for i in range(len(r)):
                ma_x, r = coordinate_descent(self.operator, ma_x, r, i)
            for i in range(len(r) - 1, -1, -1):
                ma_x, r = coordinate_descent(self.operator, ma_x, r, i)
            if jnp.linalg.norm(r) < self.tol:
                break

        z -= ma_x
        return z / jnp.max(jnp.abs(z))

    def generate_samples(
        self,
        num_samples: int,
    ) -> jnp.ndarray:
        key = random.PRNGKey(0)
        alpha = (
            random.uniform(key, ()) + 1j * random.uniform(key, ())
            if self.is_complex
            else random.uniform(key, ())
        )
        gf = ng.GridFunction(self.space)
        x_data = jnp.zeros(
            (num_samples, self.operator.shape[0]),
            dtype=jnp.complex64 if self.is_complex else jnp.float32,
        )

        for i in range(num_samples):
            gf.vec.FV().NumPy()[:] = np.random.randn(self.space.ndof)
            x_data = x_data.at[i].set(
                self.make_data(gf.vec.FV().NumPy() * alpha)
            )

        return x_data

    def get_gf(self, name: str = "gf"):
        return ng.GridFunction(self.space, name=name)


if __name__ == "__main__":
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=__MAXH__))
    dgen = HelmholtzDGen(mesh)
    print(dgen.generate_samples(10))
    print(dgen.get_gf())
