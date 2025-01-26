"""
Playground for testing the encoder-decoder architecture """

from functools import partial
import ngsolve as ng
import numpy as np
import scipy as sp
import tensorflow as tf
import plotly.express as px
from keras import ops
import keras.optimizers as opt

from geo2d import make_unit_square
from encoder import PseudoVcycle, PseudoMG
from data_gen import real_to_complex, complex_to_real, HelmholtzDGen
from losses import projected_l2_loss
from solver import symmetric_gauss_seidel, coordinate_descent


def test_vcycle_complex():
    """
    Test the Vcycle architecture with complex data
    """
    description = "Vcycle with complex data\n"
    print(description)

    tf.compat.v1.enable_eager_execution()
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.05))

    helmholtz_gen = HelmholtzDGen(
        mesh, tol=1e-1, is_complex=True, is_dirichlet=True
    )

    x_data_smooth_real = helmholtz_gen.from_smooth(1000, field="real")
    x_data_smooth_imag = helmholtz_gen.from_smooth(1000, field="imag")
    x_data_smooth_complex = helmholtz_gen.from_smooth(1000, field="complex")
    x_data_random_real = helmholtz_gen.from_random(1000, field="real")
    x_data_random_imag = helmholtz_gen.from_random(1000, field="imag")
    x_data_random_complex = helmholtz_gen.from_random(1000, field="complex")

    x_data_smooth = np.concatenate(
        (
            x_data_smooth_real,
            x_data_smooth_imag,
            x_data_smooth_complex,
        ),
        axis=0,
    )
    x_data_random = np.concatenate(
        (
            x_data_random_real,
            x_data_random_imag,
            x_data_random_complex,
        ),
        axis=0,
    )

    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = complex_to_real(x_data.T)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    x_data = tf.transpose(x_data)

    # Vcycle
    vcycle = PseudoVcycle(
        x_data.shape[1:],
        num_levels=1,
        compression_factor=5.0,
        reg_param=1e-5,
        dtype="float32",
    )

    vcycle.compile(
        optimizer="adam",
        loss="mean_absolute_error",
    )

    vcycle.fit(
        x_data,
        x_data,
        epochs=200,
        batch_size=500,
        shuffle=True,
        validation_data=(x_data, x_data),
    )

    # Summary of Vcycle
    vcycle.summary()
    vcycle.encoder.summary()
    vcycle.decoder.summary()

    # Attemp 1: Visualize residual
    x1 = x_data[0].numpy()  # real (2N, )
    x1 = real_to_complex(x1)  # complex (N, )

    x_pred = complex_to_real(x1)  # real (2N, )
    x_pred = tf.convert_to_tensor(x_pred, dtype=tf.float32)  # real (2N, )
    x_pred = tf.reshape(x_pred, (1, *x_pred.shape))  # real (1, 2N, )

    x_pred = vcycle.predict(x_pred)  # real (1, 2N, )

    x_pred = real_to_complex(x_pred[0])  # complex (N, )
    gf = helmholtz_gen.get_gf(name="res(sin(pi*x)*sin(pi*y))")
    gf.vec.FV().NumPy()[:] = x_pred  # Here

    ng.Draw(gf, mesh, "res(sin(pi*x)*sin(pi*y))")

    # Attempt 2: Encode and decode
    gf = helmholtz_gen.get_gf(name="cos(pi*x)*cos(pi*y)")
    gf.Set(ng.cos(np.pi * ng.x) * ng.cos(np.pi * ng.y))
    ng.Draw(gf, mesh, "cos(pi*x)*cos(pi*y)")

    print("Predicting...")
    x_data = np.copy(gf.vec.FV().NumPy())  # complex (N, )
    x_data = complex_to_real(x_data.T)  # real (2N, )
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)  # real (2N, )
    x_data = tf.reshape(x_data, (1, *x_data.shape))  # real (1, 2N, )
    x_pred = vcycle.predict(x_data)  # real (1, 2N, )

    gf_ed = helmholtz_gen.get_gf(name="decoded(cos(pi*x)*cos(pi*y))")
    gf_ed.vec.FV().NumPy()[:] = real_to_complex(x_pred[0])
    ng.Draw(gf_ed, mesh, "decoded(cos(pi*x)*cos(pi*y))")


def test_vcycle_real():
    """
    Test the Vcycle architecture with real data
    """
    description = "Vcycle with real data\n"
    print(description)

    tf.compat.v1.enable_eager_execution()
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.05))

    helmholtz_gen = HelmholtzDGen(
        mesh,
        tol=1e-1,
        iterations=1000,
        is_complex=False,
        is_dirichlet=True,
    )

    x_data_smooth = helmholtz_gen.from_smooth(1000, field="real")
    x_data_random = helmholtz_gen.from_random(1000, field="real")

    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    # x_data = tf.transpose(x_data)

    # Vcycle
    print("Training Vcycle...")

    vcycle = PseudoVcycle(
        x_data.shape[1:],
        num_levels=1,
        compression_factor=2.0,
        reg_param=1e-1,
        dtype="float32",
    )

    vcycle.compile(
        optimizer="adam",
        loss="mean_absolute_error",
    )

    vcycle.fit(
        x_data,
        x_data,
        epochs=200,
        batch_size=500,
        shuffle=True,
        validation_data=(x_data, x_data),
    )

    # Summary of Vcycle
    vcycle.summary()
    vcycle.encoder.summary()
    vcycle.decoder.summary()

    # Attemp 1: Visualize residual
    x1 = x_data[0].numpy()
    x_pred = tf.convert_to_tensor(x1, dtype=tf.float32)
    x_pred = tf.reshape(x_pred, (1, *x_pred.shape))

    x_pred = vcycle.predict(x_pred)

    gf = helmholtz_gen.get_gf(name="res(sin(pi*x)*sin(pi*y))")
    gf.vec.FV().NumPy()[:] = x_pred

    ng.Draw(gf, mesh, "res(sin(pi*x)*sin(pi*y))")

    # Attempt 2: Encode and decode
    gf = helmholtz_gen.get_gf(name="cos(pi*x)*cos(pi*y)")
    gf.Set(ng.cos(np.pi * ng.x) * ng.cos(np.pi * ng.y))
    ng.Draw(gf, mesh, "cos(pi*x)*cos(pi*y)")

    print("Predicting...")
    x_data = np.copy(gf.vec.FV().NumPy())
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    x_data = tf.reshape(x_data, (1, *x_data.shape))

    x_pred = vcycle.predict(x_data)

    gf_ed = helmholtz_gen.get_gf(name="decoded(cos(pi*x)*cos(pi*y))")
    gf_ed.vec.FV().NumPy()[:] = x_pred[0]
    ng.Draw(gf_ed, mesh, "decoded(cos(pi*x)*cos(pi*y))")


def test_mg_real():
    """
    Test the MG architecture
    """
    description = "MG\n"
    print(description)

    tf.compat.v1.enable_eager_execution()
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.05))

    helmholtz_gen = HelmholtzDGen(
        mesh, tol=1e-1, is_complex=False, is_dirichlet=True
    )

    x_data_smooth = helmholtz_gen.from_smooth(1000, field="real")
    x_data_random = helmholtz_gen.from_random(1000, field="real")

    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)

    matrix = helmholtz_gen.operator.todense()

    mg = PseudoMG(
        x_data.shape[1:],
        matrix=matrix,
        num_levels=1,
        compression_factor=5.0,
        reg_param=1e-6,
        dtype="float32",
    )

    mg.compile(
        optimizer="adam",
        loss=partial(
            projected_l2_loss,
            ops.transpose(mg.decoder.layers[-1].weights[0]),
        ),
    )

    mg.fit(
        x_data,
        x_data,
        epochs=200,
        batch_size=500,
        shuffle=True,
        validation_data=(x_data, x_data),
    )

    # Summary of MG
    mg.summary()

    # Attemp 1: Visualize residual
    x1 = x_data[0].numpy()
    x_pred = tf.convert_to_tensor(x1, dtype=tf.float32)
    x_pred = tf.reshape(x_pred, (1, *x_pred.shape))

    x_pred = mg.predict(x_pred)

    gf = helmholtz_gen.get_gf(name="res(sin(pi*x)*sin(pi*y))")
    gf.vec.FV().NumPy()[:] = x_pred

    ng.Draw(gf, mesh, "res(sin(pi*x)*sin(pi*y))")

    # Attempt 2: Encode and decode
    gf = helmholtz_gen.get_gf(name="cos(pi*x)*cos(pi*y)")
    gf.Set(ng.cos(np.pi * ng.x) * ng.cos(np.pi * ng.y))
    ng.Draw(gf, mesh, "cos(pi*x)*cos(pi*y)")

    print("Predicting...")
    x_data = np.copy(gf.vec.FV().NumPy())
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    x_data = tf.reshape(x_data, (1, *x_data.shape))

    x_pred = mg.predict(x_data)

    gf_ed = helmholtz_gen.get_gf(name="decoded(cos(pi*x)*cos(pi*y))")
    gf_ed.vec.FV().NumPy()[:] = x_pred[0]
    ng.Draw(gf_ed, mesh, "decoded(cos(pi*x)*cos(pi*y))")


def test_vcycle_solver():
    """
    Test the Vcycle architecture with a solver
    """
    description = (
        "Train a PseudoVcycle, assemble a two-level solver, and solve a problem\n"
        + "We implement two two-level solvers:"
        + "one with the decoder kernel and one with the encoder kernel\n"
        + "We also implement a sanity check solver\n"
    )
    print(description)

    tf.compat.v1.enable_eager_execution()

    # Set up system
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.1))
    helmholtz_gen = HelmholtzDGen(
        mesh,
        tol=1e-1,
        iterations=1000,
        is_complex=False,
        is_dirichlet=True,
    )

    x_data_smooth = helmholtz_gen.from_smooth(500, field="real")
    x_data_random = helmholtz_gen.from_random(500, field="real")

    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)

    # Vcycle
    print("Training Vcycle...")

    vcycle = PseudoVcycle(
        x_data.shape[1:],
        num_levels=1,
        compression_factor=2.0,
        reg_param=1e-3,
        dtype="float32",
    )

    vcycle.compile(
        optimizer="adam",
        loss="mean_absolute_error",
    )

    vcycle.fit(
        x_data,
        x_data,
        epochs=200,
        batch_size=100,
        shuffle=True,
        validation_data=(x_data, x_data),
    )

    # Get operators
    free_dofs = helmholtz_gen.free_dofs

    a_fine = helmholtz_gen.rest_operator
    # a = helmholtz_gen.operator
    a_ng = helmholtz_gen.ng_operator

    # decoder_kernel = vcycle.decoder.layers[-1].weights[0].numpy()  # (small, large)
    encoder_kernel = (
        vcycle.encoder.layers[-1].weights[0].numpy()
    )  # (large, small)

    TOL = 0.0
    # a_coarse_decoder = decoder_kernel @ a_fine @ decoder_kernel.T + TOL * np.eye(
    #     vcycle.inner_shape
    # )
    a_coarse_encoder = (
        encoder_kernel.T @ a_fine @ encoder_kernel
        + TOL * np.eye(vcycle.inner_shape)
    )

    # Get right-hand side
    rhs = helmholtz_gen.get_gf(name="Constant 1.0")
    rhs.Set(1.0)

    b_fine = rhs.vec.FV().NumPy().copy()

    # Make solvers
    # NGsolve solver
    sol_ng = helmholtz_gen.get_gf(name="Solution NGSolve")
    sol_ng.vec.data = (
        a_ng.Inverse(freedofs=helmholtz_gen.space.FreeDofs()) * rhs.vec
    )
    ng.Draw(sol_ng, mesh, "Solution NGSolve")

    # Symmetric Gauss-Seidel solver
    sol_sgs = helmholtz_gen.get_gf(name="Solution SGS")

    b = b_fine.copy()
    x0 = np.random.rand(len(b))

    for i in range(len(b)):
        if not free_dofs[i]:
            b[i] = 0.0

    x0 = symmetric_gauss_seidel(a_fine, x0, b, tol=1e-10, max_iter=10_000)
    sol_sgs.vec.FV().NumPy()[:] = x0
    ng.Draw(sol_sgs, mesh, "Solution SGS")

    # # Two-level solver with decoder kernel
    # sol_tl_decoder = helmholtz_gen.get_gf(name="Solution TL decoder")

    # b = b_fine.copy()
    # x_fine = np.random.rand(len(b))

    # for i in range(len(b)):
    #     if not free_dofs[i]:
    #         b[i] = 0.0

    # for _ in range(300):
    #     # Pre-smoothing
    #     r_fine = np.ravel(b - a_fine @ x_fine)
    #     e_fine = x_fine.copy()
    #     for _ in range(2):
    #         for i in range(len(e_fine)):
    #             e_fine, r_fine = coordinate_descent(a_fine, e_fine, r_fine, i)
    #     x_fine += e_fine
    #     # Coarse grid correction
    #     r_coarse = np.ravel(decoder_kernel @ r_fine)
    #     e_coarse = sp.linalg.solve(a_coarse_decoder, r_coarse)
    #     e_fine = np.ravel(decoder_kernel.T @ e_coarse)
    #     x_fine += e_fine
    #     # Post-smoothing
    #     r_fine = np.ravel(b - a_fine @ x_fine)
    #     e_fine = x_fine.copy()
    #     for _ in range(2):
    #         for i in range(len(e_fine) - 1, -1, -1):
    #             e_fine, r_fine = coordinate_descent(a_fine, e_fine, r_fine, i)
    #     x_fine += e_fine

    # sol_tl_decoder.vec.FV().NumPy()[:] = x_fine
    # ng.Draw(sol_tl_decoder, mesh, "Solution TL decoder")

    # Two-level solver with encoder kernel
    sol_tl_encoder = helmholtz_gen.get_gf(name="Solution TL encoder")

    b = b_fine.copy()
    x_fine = np.random.rand(len(b))

    for i in range(len(b)):
        if not free_dofs[i]:
            b[i] = 0.0

    for _ in range(300):
        # Pre-smoothing
        r_fine = np.ravel(b - a_fine @ x_fine)
        e_fine = x_fine.copy()
        for _ in range(2):
            for i in range(len(e_fine)):
                e_fine, r_fine = coordinate_descent(a_fine, e_fine, r_fine, i)
        x_fine += e_fine
        # Coarse grid correction
        r_coarse = np.ravel(encoder_kernel.T @ r_fine)
        # e_coarse = np.ones_like(r_coarse)
        e_coarse = sp.linalg.solve(a_coarse_encoder, r_coarse)
        e_fine = np.ravel(encoder_kernel @ e_coarse)
        x_fine += e_fine
        # Post-smoothing
        r_fine = np.ravel(b - a_fine @ x_fine)
        e_fine = x_fine.copy()
        for _ in range(2):
            for i in range(len(e_fine) - 1, -1, -1):
                e_fine, r_fine = coordinate_descent(a_fine, e_fine, r_fine, i)
        x_fine += e_fine

    sol_tl_encoder.vec.FV().NumPy()[:] = x_fine
    ng.Draw(sol_tl_encoder, mesh, "Solution TL encoder")


def test_solver():
    """
    Compare Symmetric Gauss-Seidel and native NGsolve solver
    """
    description = "Compare Symmetric Gauss-Seidel and native NGsolve solver\n"
    print(description)

    # Set up system
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.05))
    helmholtz_gen = HelmholtzDGen(
        mesh, tol=1e-1, is_complex=False, is_dirichlet=True
    )

    # Get operators
    free_dofs = helmholtz_gen.free_dofs

    a_fine = helmholtz_gen.rest_operator
    a_ng = helmholtz_gen.ng_operator

    # Get right-hand side
    rhs = helmholtz_gen.get_gf(name="Constant 1.0")
    rhs.Set(1.0)

    b_fine = rhs.vec.FV().NumPy().copy()

    # Make solvers
    # NGsolve solver
    sol_ng = helmholtz_gen.get_gf(name="Solution NGSolve")
    sol_ng.vec.data = (
        a_ng.Inverse(freedofs=helmholtz_gen.space.FreeDofs()) * rhs.vec
    )
    ng.Draw(sol_ng, mesh, "Solution NGSolve")

    # Symmetric Gauss-Seidel solver
    sol_sgs = helmholtz_gen.get_gf(name="Solution SGS")

    b = b_fine.copy()
    x0 = np.random.rand(len(b))

    for i in range(len(b)):
        if not free_dofs[i]:
            b[i] = 0.0

    x0 = symmetric_gauss_seidel(a_fine, x0, b, tol=1e-10, max_iter=10_000)
    sol_sgs.vec.FV().NumPy()[:] = x0
    ng.Draw(sol_sgs, mesh, "Solution SGS")

    l2_error = np.sqrt(
        ng.Integrate(ng.Norm(sol_ng - sol_sgs) ** 2 * ng.dx, mesh)
    )
    assert l2_error < 1e-10, f"L2 error: {l2_error}"


def test_truncate_weights():
    """
    Test the truncation of weights
    """
    description = "Truncate weights and replace the kernels\n"
    print(description)

    tf.compat.v1.enable_eager_execution()
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.1))

    helmholtz_gen = HelmholtzDGen(
        mesh,
        tol=1e-1,
        iterations=1000,
        is_complex=False,
        is_dirichlet=True,
    )

    x_data_smooth = helmholtz_gen.from_smooth(500, field="real")
    x_data_random = helmholtz_gen.from_random(500, field="real")

    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    # x_data = tf.transpose(x_data)

    # Vcycle
    print("Training Vcycle...")

    vcycle = PseudoVcycle(
        x_data.shape[1:],
        num_levels=1,
        compression_factor=2.0,
        reg_param=1e-3,
        initializer_encoder="glorot_normal",
        initializer_decoder="zeros",
        dtype="float32",
    )

    # Set up optimizer
    # 1e-2 good for glorot_normal->zeros
    # 1e-3 good for glorot_normal->glorot_normal
    # sgd = opt.SGD(learning_rate=1.0e-3, momentum=0.0)
    adam = opt.Adam(learning_rate=5.0e-3)

    vcycle.compile(
        optimizer=adam,
        # optimizer=sgd,
        loss="mean_absolute_error",
    )

    vcycle.fit(
        x_data,
        x_data,
        epochs=200,
        batch_size=100,
        shuffle=True,
        validation_data=(x_data, x_data),
    )

    # Attemp 1: Encode and decode with full weights
    gf = helmholtz_gen.get_gf(name="cos(pi*x)*cos(pi*y)")
    gf.Set(ng.cos(np.pi * ng.x) * ng.cos(np.pi * ng.y))
    ng.Draw(gf, mesh, "cos(pi*x)*cos(pi*y)")

    print("Predicting...")
    x_data = np.copy(gf.vec.FV().NumPy())
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    x_data = tf.reshape(x_data, (1, *x_data.shape))

    x_pred = vcycle.predict(x_data)

    gf_ed = helmholtz_gen.get_gf(name="decoded(cos(pi*x)*cos(pi*y))")
    gf_ed.vec.FV().NumPy()[:] = x_pred[0]
    ng.Draw(gf_ed, mesh, "decoded(cos(pi*x)*cos(pi*y))")

    # Truncate weights
    print("Truncating weights...")
    decoder_kernel = (
        vcycle.decoder.layers[-1].weights[0].numpy()
    )  # (small, large)
    encoder_kernel = (
        vcycle.encoder.layers[-1].weights[0].numpy()
    )  # (large, small)

    tol = 1e-2
    # TODO Get complexity of the operators. Cf. class notes
    decoder_kernel[np.abs(decoder_kernel) < tol] = 0.0
    encoder_kernel[np.abs(encoder_kernel) < tol] = 0.0

    vcycle.decoder.layers[-1].set_weights([decoder_kernel])
    vcycle.encoder.layers[-1].set_weights([encoder_kernel])

    # Attempt 2: Encode and decode with truncated weights
    gf_truncated = helmholtz_gen.get_gf(name="truncated(cos(pi*x)*cos(pi*y))")
    x_data_truncated = np.copy(gf.vec.FV().NumPy())
    x_data_truncated = tf.convert_to_tensor(x_data_truncated, dtype=tf.float32)
    x_data_truncated = tf.reshape(
        x_data_truncated, (1, *x_data_truncated.shape)
    )

    x_pred_truncated = vcycle.predict(x_data_truncated)
    gf_truncated.vec.FV().NumPy()[:] = x_pred_truncated[0]
    ng.Draw(gf_truncated, mesh, "truncated(cos(pi*x)*cos(pi*y))")

    # Check sparsity
    fig = px.imshow(
        np.abs(decoder_kernel), labels=dict(color="Decoder Kernel")
    )
    fig.show()

    fig = px.imshow(
        np.abs(encoder_kernel), labels=dict(color="Encoder Kernel")
    )
    fig.show()

    # Attempt 3: Set transpose of truncated encoder kernel as decoder kernel
    vcycle.decoder.layers[-1].set_weights([encoder_kernel.T])

    gf_transposed = helmholtz_gen.get_gf(
        name="transposed(cos(pi*x)*cos(pi*y))"
    )
    x_data_transposed = np.copy(gf.vec.FV().NumPy())
    x_data_transposed = tf.convert_to_tensor(
        x_data_transposed, dtype=tf.float32
    )
    x_data_transposed = tf.reshape(
        x_data_transposed, (1, *x_data_transposed.shape)
    )

    x_pred_transposed = vcycle.predict(x_data_transposed)
    gf_transposed.vec.FV().NumPy()[:] = x_pred_transposed[0]
    ng.Draw(gf_transposed, mesh, "transposed(cos(pi*x)*cos(pi*y))")


def test_check_sparsity_coarse():
    """
    Test the sparsity of the coarse operators
    """
    # print(description)

    tf.compat.v1.enable_eager_execution()

    # Set up system
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.1))
    helmholtz_gen = HelmholtzDGen(
        mesh,
        tol=1e-1,
        iterations=1000,
        is_complex=False,
        is_dirichlet=True,
    )

    x_data_smooth = helmholtz_gen.from_smooth(500, field="real")
    x_data_random = helmholtz_gen.from_random(500, field="real")

    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)

    # Vcycle
    print("Training Vcycle...")

    vcycle = PseudoVcycle(
        x_data.shape[1:],
        num_levels=1,
        compression_factor=2.0,
        reg_param=1e-3,
        dtype="float32",
    )

    vcycle.compile(
        optimizer="adam",
        loss="mean_absolute_error",
    )

    vcycle.fit(
        x_data,
        x_data,
        epochs=200,
        batch_size=100,
        shuffle=True,
        validation_data=(x_data, x_data),
    )

    # Get operators
    # free_dofs = helmholtz_gen.free_dofs

    a_fine = helmholtz_gen.rest_operator
    a = helmholtz_gen.operator
    # a_ng = helmholtz_gen.ng_operator

    decoder_kernel = (
        vcycle.decoder.layers[-1].weights[0].numpy()
    )  # (small, large)
    encoder_kernel = (
        vcycle.encoder.layers[-1].weights[0].numpy()
    )  # (large, small)

    TOL = 0.0
    a_coarse_decoder = decoder_kernel @ a @ decoder_kernel.T + TOL * np.eye(
        vcycle.inner_shape
    )
    a_coarse_encoder = encoder_kernel.T @ a @ encoder_kernel + TOL * np.eye(
        vcycle.inner_shape
    )

    a_rest_coarse_decoder = (
        decoder_kernel @ a_fine @ decoder_kernel.T
        + TOL * np.eye(vcycle.inner_shape)
    )
    a_rest_coarse_encoder = (
        encoder_kernel.T @ a_fine @ encoder_kernel
        + TOL * np.eye(vcycle.inner_shape)
    )

    a_coarse_decoder = np.abs(a_coarse_decoder)
    a_coarse_encoder = np.abs(a_coarse_encoder)
    a_rest_coarse_decoder = np.abs(a_rest_coarse_decoder)
    a_rest_coarse_encoder = np.abs(a_rest_coarse_encoder)

    fig = px.imshow(
        a_coarse_decoder, labels=dict(color="Coarse Decoder Operator")
    )
    fig.show()

    fig = px.imshow(
        a_coarse_encoder, labels=dict(color="Coarse Encoder Operator")
    )
    fig.show()

    fig = px.imshow(
        a_rest_coarse_decoder,
        labels=dict(color="Rest Coarse Decoder Operator"),
    )
    fig.show()

    fig = px.imshow(
        a_rest_coarse_encoder,
        labels=dict(color="Rest Coarse Encoder Operator"),
    )
    fig.show()


if __name__ == "__main__":
    test_vcycle_real()
    # test_vcycle_complex()
    # input("Press Enter to continue...")
    # test_mg_real()
    # test_vcycle_solver()
    # test_solver()
    # test_truncate_weights()
    # test_check_sparsity_coarse()
