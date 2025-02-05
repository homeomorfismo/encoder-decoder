"""
Data generators for different PDE problems.
"""

from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import scipy.sparse as sp
import ngsolve as ng
import pickle as pkl
from typing import Tuple, Union

# local imports
import solver as slv
from geo2d import make_unit_square

# Parameters
__MAXH__: float = 0.3
__SEED__: int = 0
__LOOP__: int = 10
__DIM__: int = 3
__NUM_SAMPLES__: int = 16

__MIN_VAL__: float = -1.0
__MAX_VAL__: float = 1.0


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

    def __str__(self):
        """
        String representation of the data generator.
        """
        return f"{self.__class__.__name__}()"

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
    def save_data_gen(self, path: str) -> None:
        """
        Save the data generator to a file.

        Args:
            path: str
        """
        pass

    def get_gf(
        self, name: str = "gf", dim: Union[None, int] = None
    ) -> ng.GridFunction:
        """
        Get a GridFunction for the data generator.

        Args:
            name: str
                Name of the GridFunction.
            dim: Union[None, int]
                Dimension of the GridFunction.

        Returns:
            GridFunction
                GridFunction object.
        """
        return (
            ng.GridFunction(self.space, name=name)
            if dim is None
            else ng.GridFunction(self.space, name=name, multidim=dim)
        )

    def assemble(self, *args) -> None:
        """
        Assemble the forms in the data generator.

        Args:
            args: list
                List of forms to assemble.
        """
        for form in args:
            with ng.TaskManager():
                try:
                    form.Assemble()
                except Exception as e:
                    print(
                        f"\t->Unable to assemble {form}. Increasing heap size."
                        f"\nError: {e}"
                    )
                    ng.SetHeapSize(int(1e9))
                    form.Assemble()
                finally:
                    pass
        print("\t->All forms assembled.")

    def rnd_shape(
        self,
        key: jax.random.PRNGKey,
        shape: Tuple[int, ...],
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
        """
        Generate a random array with a given shape.

        Args:
            key: jax.random.PRNGKey
                PRNG key for random number generation.
            shape: Tuple[int, ...]
                Shape of the random array.

        Returns:
            Tuple[jax.random.PRNGKey, jnp.ndarray]
                New PRNG key and random array.
        """
        new_key, sub_key = jax.random.split(key)
        return new_key, jax.random.uniform(
            sub_key, shape, minval=__MIN_VAL__, maxval=__MAX_VAL__
        )


# Data generators
class BasicConvDiffDataGen(DataGenerator):
    """
    Data generator for a basic convection-diffusion problem.
    """

    def __init__(
        self,
        mesh: ng.Mesh,
        tol: float = 1e-3,
        order: int = 1,
        iterations: int = 1_000,
        is_complex: bool = True,
        is_dirichlet: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the data generator.
        INPUTS:
            mesh: ng.Mesh
                NGSolve mesh object.
            tol: float
                Tolerance for the iterative solver.
            order: int
                Order of the finite element space.
            iterations: int
                Number of iterations for the iterative solver.
            is_complex: bool
                Flag for complex data.
            is_dirichlet: bool
                Flag for Dirichlet boundary conditions.
        """
        super().__init__()
        matrix_coeff = ng.CF((1.0, 0.0, 0.0, 1.0), dims=(2, 2))
        vector_coeff = ng.CF((0.0, 0.0))
        scalar_coeff = ng.CF(1.0)
        if is_dirichlet:
            fes = ng.H1(
                mesh,
                order=order,
                complex=is_complex,
                dirichlet="boundary",
                autoupdate=True,
            )
        else:
            fes = ng.H1(mesh, order=order, complex=is_complex, autoupdate=True)
        u, v = fes.TnT()
        a = ng.BilinearForm(fes)
        a += matrix_coeff * ng.grad(u) * ng.grad(v) * ng.dx
        a += vector_coeff * ng.grad(u) * v * ng.dx
        a += scalar_coeff * u * v * ng.dx
        self.assemble(a)

        self.space = fes
        a_row, a_col, a_val = a.mat.COO()
        self.operator = jnp.array(
            sp.csr_matrix(
                (a_val, (a_row, a_col)),
                shape=(a.mat.height, a.mat.width),
            ).todense()
        )
        self.free_dofs = jnp.array(list(fes.FreeDofs()), dtype=jnp.bool_)
        self.rest_operator = self.operator.at[~self.free_dofs, :].set(0)
        self.rest_operator = self.rest_operator.at[:, ~self.free_dofs].set(0)
        self.rest_operator = self.rest_operator.at[
            jnp.diag_indices_from(self.rest_operator)
        ].set(self.operator.diagonal())

        self.tol = tol
        self.iterations = iterations
        self.is_complex = is_complex

        if debug:
            print(
                f"Debugging {self.__class__.__name__} data generator:\n"
                f"\t->operator:\n{self.operator}\n"
                f"\t->rest operator:\n{self.rest_operator}\n"
                f"\t->free dofs: {self.free_dofs}\n"
            )

        print(f"\t->Data generator {self.__class__.__name__} initialized.")

    def __str__(self):
        desc = (
            f"Space: {self.space}\n"
            f"Operator shape: {self.operator.shape}\n"
            f"Rest operator shape: {self.rest_operator.shape}\n"
            f"Tolerance: {self.tol}\n"
            f"Iterations: {self.iterations}\n"
            f"Complex: {self.is_complex}"
        )
        return f"{super().__str__()}\n{desc}"

    def make_data(self, x: jnp.ndarray, use_rest: bool = False) -> jnp.ndarray:
        """
        Apply symmetric Gauss-Seidel with prescibed tolerance
        and maximum number of iterations.
        """
        z = x.copy()
        b = jnp.dot(self.operator, z)
        ma_x = jnp.ones_like(b)
        ma_x = slv.forward_gauss_seidel(
            self.rest_operator if use_rest else self.operator,
            ma_x,
            b,
            tol=self.tol,
            max_iter=self.iterations,
        )
        ma_x = slv.backward_gauss_seidel(
            self.rest_operator if use_rest else self.operator,
            ma_x,
            b,
            tol=self.tol,
            max_iter=self.iterations,
        )
        z -= ma_x
        return z / jnp.max(jnp.abs(z))

    def generate_random_samples(
        self,
        num_samples: int,
        key: jax.random.PRNGKey = jax.random.PRNGKey(__SEED__),
        use_rest: bool = False,
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
        gf = ng.GridFunction(self.space)
        x_data = jnp.zeros(
            (num_samples, self.operator.shape[0]),
            dtype=gf.vec.FV().NumPy().dtype,
        )

        for i in range(num_samples):
            key, rnd_values = self.rnd_shape(key, gf.vec.FV().NumPy().shape)
            gf.vec.FV().NumPy()[:] = rnd_values
            x_data = x_data.at[i].set(
                self.make_data(gf.vec.FV().NumPy()[:], use_rest)
            )

        return key, x_data

    def generate_random_nbc_samples(
        self,
        num_samples: int,
        key: jax.random.PRNGKey = jax.random.PRNGKey(__SEED__),
        use_rest: bool = False,
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
        gf = ng.GridFunction(self.space)
        x_data = jnp.zeros(
            (num_samples, self.operator.shape[0]),
            dtype=gf.vec.FV().NumPy().dtype,
        )

        for i in range(num_samples):
            key, rnd_values = self.rnd_shape(key, gf.vec.FV().NumPy().shape)
            gf.vec.FV().NumPy()[:] = rnd_values
            gf.vec.FV().NumPy()[self.free_dofs] = 0.0
            x_data = x_data.at[i].set(
                self.make_data(gf.vec.FV().NumPy()[:], use_rest)
            )
        x_data = x_data.at[:, ~self.free_dofs].set(0.0)
        return key, x_data

    def generate_sinusoidal_samples(
        self,
        num_samples: int,
        key: jax.random.PRNGKey = jax.random.PRNGKey(__SEED__),
        use_rest: bool = False,
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
        gf = ng.GridFunction(self.space)
        x_data = jnp.zeros(
            (num_samples, self.operator.shape[0]),
            dtype=gf.vec.FV().NumPy().dtype,
        )
        for i in range(num_samples):
            key, alpha = self.rnd_shape(key, ())
            alpha *= 2.0 * jnp.pi
            key, beta = self.rnd_shape(key, ())
            beta *= 2.0 * jnp.pi
            key, gamma = self.rnd_shape(key, ())
            key, delta = self.rnd_shape(key, ())
            gamma = gamma + delta * 1j if self.is_complex else gamma
            gf.Set(
                gamma
                + (beta + gamma) * ng.sin(alpha * i * ng.x)
                + (alpha - gamma) * ng.cos(beta * i * ng.y)
            )
            x_data = x_data.at[i].set(
                self.make_data(gf.vec.FV().NumPy()[:], use_rest)
            )

        return key, x_data

    def generate_sinuoidal_samples_nbc(
        self,
        num_samples: int,
        key: jax.random.PRNGKey = jax.random.PRNGKey(__SEED__),
        use_rest: bool = False,
    ) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
        gf = ng.GridFunction(self.space)
        x_data = jnp.zeros(
            (num_samples, self.operator.shape[0]),
            dtype=gf.vec.FV().NumPy().dtype,
        )
        for i in range(num_samples):
            key, alpha = self.rnd_shape(key, ())
            alpha *= 2.0 * jnp.pi
            key, beta = self.rnd_shape(key, ())
            beta *= 2.0 * jnp.pi
            key, gamma = self.rnd_shape(key, ())
            key, delta = self.rnd_shape(key, ())
            gamma = gamma + delta * 1j if self.is_complex else gamma
            gf.Set(
                gamma
                + (beta + gamma) * ng.sin(alpha * i * ng.x)
                + (alpha - gamma) * ng.cos(beta * i * ng.y)
            )
            gf.vec.FV().NumPy()[self.free_dofs] = 0.0
            x_data = x_data.at[i].set(
                self.make_data(gf.vec.FV().NumPy()[:], use_rest)
            )
        x_data = x_data.at[:, ~self.free_dofs].set(0.0)

        return key, x_data

    def generate_samples(
        self,
        num_samples: int,
        key: jax.random.PRNGKey = jax.random.PRNGKey(__SEED__),
        use_rest: bool = False,
    ) -> jnp.ndarray:
        key, random_samples = self.generate_random_samples(
            num_samples // 4, key, use_rest
        )
        key, nbc_samples = self.generate_random_nbc_samples(
            num_samples // 4, key, use_rest
        )
        key, sinusoidal_samples = self.generate_sinusoidal_samples(
            num_samples // 4, key, use_rest
        )
        key, sinusoidal_nbc_samples = self.generate_sinuoidal_samples_nbc(
            num_samples // 4, key, use_rest
        )
        x_data = jnp.vstack(
            (
                random_samples,
                nbc_samples,
                sinusoidal_samples,
                sinusoidal_nbc_samples,
            )
        )
        return x_data

    def save_data_gen(self, path: str):
        kwargs = {
            "mesh": self.space.mesh.ngmesh,
            "tol": self.tol,
            "order": self.space.order,
            "iterations": self.iterations,
            "is_complex": self.is_complex,
            "is_dirichlet": self.space.dirichlet,
        }
        with open(path, "wb") as f:
            pkl.dump(kwargs, f)
