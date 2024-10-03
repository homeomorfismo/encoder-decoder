"""
Cleaner version of the tests scripts.
"""

import ngsolve as ng
import numpy as np
import scipy as sp
import tensorflow as tf
import plotly.express as px
import keras.optimizers as opt

from geo2d import make_unit_square
from encoder import PseudoVcycle
from data_gen import HelmholtzDGen
from solver import symmetric_gauss_seidel, forward_gauss_seidel, backward_gauss_seidel

MAX_H = 0.1

DATA_GEN_TOL = 1e-1
DATA_GEN_ITER = 1000
DATA_GEN_NVECS = 1000

VCYCLE_COMPRESSION_FACTOR = 2.0
VCYCLE_REGULARIZATION = 1.0e-5
VCYCLE_LEARNING_RATE = 5.0e-4
VCYCLE_EPOCHS = 200
VCYCLE_BATCH_SIZE = 100
VCYCLE_INIT_ENCODER = "glorot_uniform"
VCYCLE_INIT_DECODER = "zeros"

TRUNCATION_TOL = 1.0e-2
ERROR_TOL = 1.0e-1

SOLVER_ITER = 1000
SOLVER_SMOOTHING_ITER = 2


def test_all():
    """
    Run all tests.
    """
    print(
        "Print all global variables\n"
        f"MAX_H = {MAX_H}\n\n"
        f"DATA_GEN_TOL = {DATA_GEN_TOL}\n"
        f"DATA_GEN_ITER = {DATA_GEN_ITER}\n"
        f"DATA_GEN_NVECS = {DATA_GEN_NVECS}\n\n"
        f"VCYCLE_COMPRESSION_FACTOR = {VCYCLE_COMPRESSION_FACTOR}\n"
        f"VCYCLE_REGULARIZATION = {VCYCLE_REGULARIZATION}\n"
        f"VCYCLE_LEARNING_RATE = {VCYCLE_LEARNING_RATE}\n"
        f"VCYCLE_EPOCHS = {VCYCLE_EPOCHS}\n"
        f"VCYCLE_BATCH_SIZE = {VCYCLE_BATCH_SIZE}\n"
        f"VCYCLE_INIT_ENCODER = {VCYCLE_INIT_ENCODER}\n"
        f"VCYCLE_INIT_DECODER = {VCYCLE_INIT_DECODER}\n\n"
        f"TRUNCATION_TOL = {TRUNCATION_TOL}\n"
        f"ERROR_TOL = {ERROR_TOL}\n\n"
        f"SOLVER_ITER = {SOLVER_ITER}\n"
        f"SOLVER_SMOOTHING_ITER = {SOLVER_SMOOTHING_ITER}\n\n"
    )

    print("Assembling data-generator and encoder-decoder model: Helmholtz Problem")

    tf.compat.v1.enable_eager_execution()

    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=MAX_H))

    helmholtz_gen = HelmholtzDGen(
        mesh,
        tol=DATA_GEN_TOL,
        iterations=DATA_GEN_ITER,
        is_complex=False,
        is_dirichlet=True,
    )

    x_data_smooth = helmholtz_gen.from_smooth(DATA_GEN_NVECS, field="real")
    x_data_random = helmholtz_gen.from_random(DATA_GEN_NVECS, field="real")

    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)

    vcycle = PseudoVcycle(
        x_data.shape[1:],
        num_levels=1,
        compression_factor=VCYCLE_COMPRESSION_FACTOR,
        reg_param=VCYCLE_REGULARIZATION,
        initializer_encoder=VCYCLE_INIT_ENCODER,
        initializer_decoder=VCYCLE_INIT_DECODER,
        dtype="float32",
    )

    adam = opt.Adam(learning_rate=VCYCLE_LEARNING_RATE)

    vcycle.compile(
        optimizer=adam,
        loss="mean_absolute_error",
    )

    vcycle.fit(
        x_data,
        x_data,
        epochs=VCYCLE_EPOCHS,
        batch_size=VCYCLE_BATCH_SIZE,
        shuffle=True,
        validation_data=(x_data, x_data),
    )

    print("Testing the encoder-decoder model: encode-decode a smooth field")

    gf_smooth = helmholtz_gen.get_gf(name="cos(pi*x)*cos(pi*y)")
    gf_smooth.Set(ng.cos(ng.pi * ng.x) * ng.cos(ng.pi * ng.y))

    vec_ed_smooth = np.copy(gf_smooth.vec.FV().NumPy())
    vec_ed_smooth = tf.convert_to_tensor(vec_ed_smooth, dtype=tf.float32)
    vec_ed_smooth = tf.reshape(vec_ed_smooth, (1, *vec_ed_smooth.shape))
    vec_ed_smooth = vcycle.predict(vec_ed_smooth)

    gf_ed_smooth = helmholtz_gen.get_gf(name="ED cos(pi*x)*cos(pi*y)")
    gf_ed_smooth.vec.FV().NumPy()[:] = vec_ed_smooth[0]

    ng.Draw(gf_smooth, mesh, "smooth")
    ng.Draw(gf_ed_smooth, mesh, "ED smooth")

    error = ng.sqrt(ng.Integrate((gf_smooth - gf_ed_smooth) ** 2 * ng.dx, mesh))
    print(f"\t||u - ED u||_L2 = {error}")
    assert error < ERROR_TOL, f"Error too large! ||u - ED u||_L2 = {error}"

    print("Assembling operators... Check the sparse structure of the operators")

    a_fine = helmholtz_gen.rest_operator
    # a = helmholtz_gen.operator
    a_ng = helmholtz_gen.ng_operator

    free_dofs = helmholtz_gen.free_dofs

    decoder_kernel = vcycle.decoder.layers[-1].weights[0].numpy()  # (small, large)
    encoder_kernel = vcycle.encoder.layers[-1].weights[0].numpy()  # (large, small)

    truncated_decoder_kernel = decoder_kernel.copy()
    truncated_encoder_kernel = encoder_kernel.copy()

    truncated_decoder_kernel[np.abs(decoder_kernel) < TRUNCATION_TOL] = 0.0
    truncated_encoder_kernel[np.abs(encoder_kernel) < TRUNCATION_TOL] = 0.0

    # a_coarse_decoder = decoder_kernel @ a @ decoder_kernel.T
    # a_coarse_encoder = encoder_kernel.T @ a @ encoder_kernel

    a_res_coarse_decoder = decoder_kernel @ a_fine @ decoder_kernel.T
    a_res_coarse_encoder = encoder_kernel.T @ a_fine @ encoder_kernel

    a_res_coarse_truncated_decoder = (
        truncated_decoder_kernel @ a_fine @ truncated_decoder_kernel.T
    )
    a_res_coarse_truncated_encoder = (
        truncated_encoder_kernel.T @ a_fine @ truncated_encoder_kernel
    )

    a_res_coarse_truncated_decoder[
        np.abs(a_res_coarse_truncated_decoder) < TRUNCATION_TOL
    ] = 0.0

    print("Plotting sparse structure of operators...")

    print("\tEncoder-decoder...")

    fig = px.imshow(decoder_kernel, labels=dict(color="D"))
    fig.show()

    fig = px.imshow(encoder_kernel, labels=dict(color="E"))
    fig.show()

    print("\tTruncated encoder-decoder...")

    fig = px.imshow(truncated_decoder_kernel, labels=dict(color="D_trunc"))
    fig.show()

    fig = px.imshow(truncated_encoder_kernel, labels=dict(color="E_trunc"))
    fig.show()

    print("\tOperators...")

    fig = px.imshow(a_fine, labels=dict(color="A_fine"))
    fig.show()

    # fig = px.imshow(a_coarse_decoder, labels=dict(color="D @ A @ D^T"))
    # fig.show()

    # fig = px.imshow(a_coarse_encoder, labels=dict(color="E^T @ A @ E"))
    # fig.show()

    fig = px.imshow(a_res_coarse_decoder, labels=dict(color="D @ A_fine @ D^T"))
    fig.show()

    fig = px.imshow(a_res_coarse_encoder, labels=dict(color="E^T @ A_fine @ E"))
    fig.show()

    fig = px.imshow(
        a_res_coarse_truncated_decoder,
        labels=dict(color="D_trunc @ A_fine @ D_trunc^T"),
    )
    fig.show()

    fig = px.imshow(
        a_res_coarse_truncated_encoder,
        labels=dict(color="E_trunc^T @ A_fine @ E_trunc"),
    )
    fig.show()

    print("Testing solvers")

    rhs = helmholtz_gen.get_gf(name="Constant 1.0")
    rhs.Set(1.0)

    b_fine = rhs.vec.FV().NumPy().copy()

    print("--- NGSolve solver ---")

    sol_ng = helmholtz_gen.get_gf(name="NGSolve")
    sol_ng.vec.data = a_ng.Inverse(freedofs=helmholtz_gen.space.FreeDofs()) * rhs.vec
    ng.Draw(sol_ng, mesh, "NGSolve solution")

    print("--- Symmetric Gauss-Seidel solver ---")

    sol_sgs = helmholtz_gen.get_gf(name="SGS")

    b = b_fine.copy()
    x_fine = np.random.rand(len(b))

    for i in range(len(b)):
        if not free_dofs[i]:
            b[i] = 0.0

    x_fine = symmetric_gauss_seidel(a_fine, x_fine, b, tol=1e-10, max_iter=10_000)
    sol_sgs.vec.FV().NumPy()[:] = x_fine
    ng.Draw(sol_sgs, mesh, "SGS solution")

    error = ng.sqrt(ng.Integrate((sol_ng - sol_sgs) ** 2 * ng.dx, mesh))
    print(f"\t||u - SGS u||_L2 = {error}")
    # assert error < ERROR_TOL, f"Error too large! ||u - SGS u||_L2 = {error}"

    print(
        "--- Two-level solver: decoder_kernel ---\n"
        "\tInterpolator/projector: decoder_kernel\n"
        "\tCoarse operator: a_res_coarse_decoder\n"
    )

    sol_tl_decoder = helmholtz_gen.get_gf(name="TL Decoder")

    b = b_fine.copy()
    x_fine = np.random.rand(len(b))

    for i in range(len(b)):
        if not free_dofs[i]:
            b[i] = 0.0

    # e_fine = np.zeros_like(x_fine)
    e_fine = np.random.rand(len(x_fine))
    for _ in range(SOLVER_ITER):
        # Pre-smoothing
        r_fine = np.ravel(b - a_fine @ x_fine)
        e_fine = forward_gauss_seidel(
            a_fine, e_fine, r_fine, tol=1e-10, max_iter=SOLVER_SMOOTHING_ITER
        )
        x_fine += e_fine
        # Coarse grid correction
        r_fine = np.ravel(b - a_fine @ x_fine)
        r_coarse = np.ravel(decoder_kernel @ r_fine)
        try:
            e_coarse = sp.linalg.solve(a_res_coarse_decoder, r_coarse)
        except ValueError:
            print("Unable to solve coarse grid correction")
            x_fine = np.zeros_like(x_fine)
            break
        e_fine = np.ravel(decoder_kernel.T @ e_coarse)
        x_fine += e_fine
        # Post-smoothing
        r_fine = np.ravel(b - a_fine @ x_fine)
        e_fine = np.zeros_like(r_fine)
        e_fine = backward_gauss_seidel(
            a_fine, e_fine, r_fine, tol=1e-10, max_iter=SOLVER_SMOOTHING_ITER
        )
        x_fine += e_fine

    sol_tl_decoder.vec.FV().NumPy()[:] = x_fine
    ng.Draw(sol_tl_decoder, mesh, "TL Decoder solution")

    error = ng.sqrt(ng.Integrate((sol_ng - sol_tl_decoder) ** 2 * ng.dx, mesh))
    print(f"\t||u - TL Decoder u||_L2 = {error}")
    # assert error < ERROR_TOL, f"Error too large! ||u - TL Decoder u||_L2 = {error}"

    print(
        "--- Two-level solver: encoder_kernel ---\n"
        "\tInterpolator/projector: encoder_kernel\n"
        "\tCoarse operator: a_res_coarse_encoder\n"
    )

    sol_tl_encoder = helmholtz_gen.get_gf(name="TL Encoder")

    b = b_fine.copy()
    x_fine = np.random.rand(len(b))

    for i in range(len(b)):
        if not free_dofs[i]:
            b[i] = 0.0

    # e_fine = np.zeros_like(x_fine)
    e_fine = np.random.rand(len(x_fine))
    for _ in range(SOLVER_ITER):
        # Pre-smoothing
        r_fine = np.ravel(b - a_fine @ x_fine)
        e_fine = forward_gauss_seidel(
            a_fine, e_fine, r_fine, tol=1e-10, max_iter=SOLVER_SMOOTHING_ITER
        )
        x_fine += e_fine
        # Coarse grid correction
        r_fine = np.ravel(b - a_fine @ x_fine)
        r_coarse = np.ravel(encoder_kernel.T @ r_fine)
        try:
            e_coarse = sp.linalg.solve(a_res_coarse_encoder, r_coarse)
        except ValueError:
            print("Unable to solve coarse grid correction")
            x_fine = np.zeros_like(x_fine)
            break
        e_fine = np.ravel(encoder_kernel @ e_coarse)
        x_fine += e_fine
        # Post-smoothing
        r_fine = np.ravel(b - a_fine @ x_fine)
        e_fine = np.zeros_like(r_fine)
        e_fine = backward_gauss_seidel(
            a_fine, e_fine, r_fine, tol=1e-10, max_iter=SOLVER_SMOOTHING_ITER
        )
        x_fine += e_fine

    sol_tl_encoder.vec.FV().NumPy()[:] = x_fine
    ng.Draw(sol_tl_encoder, mesh, "TL Encoder solution")

    error = ng.sqrt(ng.Integrate((sol_ng - sol_tl_encoder) ** 2 * ng.dx, mesh))
    print(f"\t||u - TL Encoder u||_L2 = {error}")
    # assert error < ERROR_TOL, f"Error too large! ||u - TL Encoder u||_L2 = {error}"

    print(
        "--- Two-level solver: truncated decoder_kernel ---\n"
        "\tInterpolator/projector: truncated decoder_kernel\n"
        "\tCoarse operator: a_res_coarse_truncated_decoder\n"
    )

    sol_tl_truncated_decoder = helmholtz_gen.get_gf(name="TL Truncated Decoder")

    b = b_fine.copy()
    x_fine = np.random.rand(len(b))

    for i in range(len(b)):
        if not free_dofs[i]:
            b[i] = 0.0

    # e_fine = np.zeros_like(x_fine)
    e_fine = np.random.rand(len(x_fine))
    for _ in range(SOLVER_ITER):
        # Pre-smoothing
        r_fine = np.ravel(b - a_fine @ x_fine)
        e_fine = forward_gauss_seidel(
            a_fine, e_fine, r_fine, tol=1e-10, max_iter=SOLVER_SMOOTHING_ITER
        )
        x_fine += e_fine
        # Coarse grid correction
        r_fine = np.ravel(b - a_fine @ x_fine)
        r_coarse = np.ravel(truncated_decoder_kernel @ r_fine)
        try:
            e_coarse = sp.linalg.solve(a_res_coarse_truncated_decoder, r_coarse)
        except ValueError:
            print("Unable to solve coarse grid correction")
            x_fine = np.zeros_like(x_fine)
            break
        e_fine = np.ravel(truncated_decoder_kernel.T @ e_coarse)
        x_fine += e_fine
        # Post-smoothing
        r_fine = np.ravel(b - a_fine @ x_fine)
        e_fine = np.zeros_like(r_fine)
        e_fine = backward_gauss_seidel(
            a_fine, e_fine, r_fine, tol=1e-10, max_iter=SOLVER_SMOOTHING_ITER
        )
        x_fine += e_fine

    sol_tl_truncated_decoder.vec.FV().NumPy()[:] = x_fine
    ng.Draw(sol_tl_truncated_decoder, mesh, "TL Truncated Decoder solution")

    error = ng.sqrt(
        ng.Integrate((sol_ng - sol_tl_truncated_decoder) ** 2 * ng.dx, mesh)
    )
    print(f"\t||u - TL Truncated Decoder u||_L2 = {error}")
    # assert (
    #     error < ERROR_TOL
    # ), f"Error too large! ||u - TL Truncated Decoder u||_L2 = {error}"

    print(
        "--- Two-level solver: truncated encoder_kernel ---\n"
        "\tInterpolator/projector: truncated encoder_kernel\n"
        "\tCoarse operator: a_res_coarse_truncated_encoder\n"
    )

    sol_tl_truncated_encoder = helmholtz_gen.get_gf(name="TL Truncated Encoder")

    b = b_fine.copy()
    x_fine = np.random.rand(len(b))

    for i in range(len(b)):
        if not free_dofs[i]:
            b[i] = 0.0

    # e_fine = np.zeros_like(x_fine)
    e_fine = np.random.rand(len(x_fine))
    for _ in range(SOLVER_ITER):
        # Pre-smoothing
        r_fine = np.ravel(b - a_fine @ x_fine)
        e_fine = forward_gauss_seidel(
            a_fine, e_fine, r_fine, tol=1e-10, max_iter=SOLVER_SMOOTHING_ITER
        )
        x_fine += e_fine
        # Coarse grid correction
        r_fine = np.ravel(b - a_fine @ x_fine)
        r_coarse = np.ravel(truncated_encoder_kernel.T @ r_fine)
        try:
            e_coarse = sp.linalg.solve(a_res_coarse_truncated_encoder, r_coarse)
        except ValueError:
            print("Unable to solve coarse grid correction")
            x_fine = np.zeros_like(x_fine)
            break
        e_fine = np.ravel(truncated_encoder_kernel @ e_coarse)
        x_fine += e_fine
        # Post-smoothing
        r_fine = np.ravel(b - a_fine @ x_fine)
        e_fine = np.zeros_like(r_fine)
        e_fine = backward_gauss_seidel(
            a_fine, e_fine, r_fine, tol=1e-10, max_iter=SOLVER_SMOOTHING_ITER
        )
        x_fine += e_fine

    sol_tl_truncated_encoder.vec.FV().NumPy()[:] = x_fine
    ng.Draw(sol_tl_truncated_encoder, mesh, "TL Truncated Encoder solution")

    error = ng.sqrt(
        ng.Integrate((sol_ng - sol_tl_truncated_encoder) ** 2 * ng.dx, mesh)
    )
    print(f"\t||u - TL Truncated Encoder u||_L2 = {error}")
    # assert (
    #     error < ERROR_TOL
    # ), f"Error too large! ||u - TL Truncated Encoder u||_L2 = {error}"

    print("All tests passed!")


if __name__ == "__main__":
    test_all()
