"""
Some basic matrices from standard PDEs
"""

from ngsolve import (
    TaskManager,
    SetHeapSize,
    CoefficientFunction,
    grad,
    dx,
    H1,
    BilinearForm,
    Mesh,
    BND,
    Draw,
    GridFunction,
)
import numpy as np
from geo2d import make_unit_square


def assemble(*args) -> None:
    """
    Assemble the forms
    """
    for form in args:
        with TaskManager():
            try:
                form.Assemble()
            except Exception as e:
                print(
                    f"Unable to assemble {form}. Increasing heap size."
                    f"\nError: {e}"
                )
                SetHeapSize(int(1e9))
                form.Assemble()
            finally:
                pass


def convection_diffusion(
    mesh,
    matrix_coeff=None,
    vector_coeff=None,
    scalar_coeff=None,
    order: int = 1,
    is_complex: bool = True,
    is_dirichlet: bool = True,
):
    """
    Assemble the convection-diffusion equation
    """
    assert matrix_coeff is not None, "Matrix coefficient must be provided"
    assert vector_coeff is not None, "Vector coefficient must be provided"
    assert scalar_coeff is not None, "Scalar coefficient must be provided"
    if is_dirichlet:
        fes = H1(
            mesh,
            order=order,
            complex=is_complex,
            dirichlet="boundary",
            autoupdate=True,
        )
    else:
        fes = H1(mesh, order=order, complex=is_complex, autoupdate=True)
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += matrix_coeff * grad(u) * grad(v) * dx
    a += vector_coeff * grad(u) * v * dx
    a += scalar_coeff * u * v * dx
    m = BilinearForm(fes)
    m += u * v * dx
    return a, m, fes


def test_convection_diffusion():
    mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.1))
    matrix = CoefficientFunction((1.0, 0.0, 0.0, 1.0), dims=(2, 2))
    vector = CoefficientFunction((0.0, 0.0))
    scalar = CoefficientFunction(0.0)
    a_mat, m_mat, space = convection_diffusion(
        mesh,
        matrix_coeff=matrix,
        vector_coeff=vector,
        scalar_coeff=scalar,
        order=1,
        is_complex=True,
    )
    print(
        f"Number of DoFs: {space.ndof}\n"
        f"Number of elements: {mesh.ne}\n"
        f"\tfes: {space}\n"
        f"\ta: {a_mat}\n"
        f"\tm: {m_mat}\n"
    )


def test_vectors():
    """
    Test vector coefficients
    """
    mesh = Mesh(make_unit_square().GenerateMesh(maxh=0.1))
    matrix = CoefficientFunction((1.0, 0.0, 0.0, 1.0), dims=(2, 2))
    vector = CoefficientFunction((0.0, 0.0))
    scalar = CoefficientFunction(0.0)
    _, _, space = convection_diffusion(
        mesh,
        matrix_coeff=matrix,
        vector_coeff=vector,
        scalar_coeff=scalar,
        order=1,
        is_complex=True,
    )

    free_dofs = np.array(list(space.FreeDofs()))

    gf = GridFunction(space)
    man_gf = GridFunction(space)
    ex_gf = GridFunction(space)

    gf.Set(1.0, BND)
    Draw(gf, mesh, "gf_ngsolve")

    x = np.zeros_like(man_gf.vec.FV().NumPy())
    for i in range(len(x)):
        if not free_dofs[i]:  # BND
            x[i] = 1.0
    man_gf.vec.data.FV().NumPy()[:] = x
    Draw(man_gf, mesh, "gf_manual")

    y = np.ones_like(ex_gf.vec.FV().NumPy())
    for i in range(len(y)):
        if not free_dofs[i]:  # BND
            y[i] = 0.0
    ex_gf.vec.data.FV().NumPy()[:] = y
    Draw(ex_gf, mesh, "gf_example")

    print(f"Free dofs: {free_dofs}")
    print(f"Vector coefficient: {gf.vec.FV().NumPy()}")


if __name__ == "__main__":
    test_convection_diffusion()
    test_vectors()
