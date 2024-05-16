"""
Playground for testing the encoder-decoder architecture
"""

from functools import partial
import ngsolve as ng
import numpy as np
import scipy as sp
import tensorflow as tf
from keras import ops

from geo2d import make_unit_square
from encoder import PseudoVcycle, PseudoMG
from data_gen import real_to_complex, complex_to_real, LaplaceDGen

# from metrics import L2ErrorMetric
from losses import projected_l2_loss

# from fes import assemble, convection_diffusion


def test_vcycle_complex():
    """
    Test the Vcycle architecture with complex data
    """
    description = "Vcycle with complex data\n"
    print(description)

    tf.compat.v1.enable_eager_execution()
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.05))

    laplace_gen = LaplaceDGen(mesh, tol=1e-1, is_complex=True)

    x_data_smooth_real = laplace_gen.from_smooth(1000, field="real")
    x_data_smooth_imag = laplace_gen.from_smooth(1000, field="imag")
    x_data_smooth_complex = laplace_gen.from_smooth(1000, field="complex")
    x_data_random_real = laplace_gen.from_random(1000, field="real")
    x_data_random_imag = laplace_gen.from_random(1000, field="imag")
    x_data_random_complex = laplace_gen.from_random(1000, field="complex")

    x_data_smooth = np.concatenate(
        (x_data_smooth_real, x_data_smooth_imag, x_data_smooth_complex), axis=0
    )
    x_data_random = np.concatenate(
        (x_data_random_real, x_data_random_imag, x_data_random_complex), axis=0
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
        # metrics=[L2ErrorMetric(space=laplace_gen.ngsolve_operator.space)],
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
    gf = laplace_gen.get_gf(name="res(sin(pi*x)*sin(pi*y))")
    gf.vec.FV().NumPy()[:] = x_pred  # Here

    ng.Draw(gf, mesh, "res(sin(pi*x)*sin(pi*y))")

    # Attempt 2: Encode and decode
    gf = laplace_gen.get_gf(name="cos(pi*x)*cos(pi*y)")
    gf.Set(ng.cos(np.pi * ng.x) * ng.cos(np.pi * ng.y))
    ng.Draw(gf, mesh, "cos(pi*x)*cos(pi*y)")

    print("Predicting...")
    x_data = np.copy(gf.vec.FV().NumPy())  # complex (N, )
    x_data = complex_to_real(x_data.T)  # real (2N, )
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)  # real (2N, )
    x_data = tf.reshape(x_data, (1, *x_data.shape))  # real (1, 2N, )
    x_pred = vcycle.predict(x_data)  # real (1, 2N, )

    gf_ed = laplace_gen.get_gf(name="decoded(cos(pi*x)*cos(pi*y))")
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

    laplace_gen = LaplaceDGen(mesh, tol=1e-1, is_complex=False)

    x_data_smooth = laplace_gen.from_smooth(1000, field="real")
    x_data_random = laplace_gen.from_random(1000, field="real")

    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    # x_data = tf.transpose(x_data)

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
        # metrics=[L2ErrorMetric(space=laplace_gen.ngsolve_operator.space)],
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

    gf = laplace_gen.get_gf(name="res(sin(pi*x)*sin(pi*y))")
    gf.vec.FV().NumPy()[:] = x_pred

    ng.Draw(gf, mesh, "res(sin(pi*x)*sin(pi*y))")

    # Attempt 2: Encode and decode
    gf = laplace_gen.get_gf(name="cos(pi*x)*cos(pi*y)")
    gf.Set(ng.cos(np.pi * ng.x) * ng.cos(np.pi * ng.y))
    ng.Draw(gf, mesh, "cos(pi*x)*cos(pi*y)")

    print("Predicting...")
    x_data = np.copy(gf.vec.FV().NumPy())
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    x_data = tf.reshape(x_data, (1, *x_data.shape))

    x_pred = vcycle.predict(x_data)

    gf_ed = laplace_gen.get_gf(name="decoded(cos(pi*x)*cos(pi*y))")
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

    laplace_gen = LaplaceDGen(mesh, tol=1e-1, is_complex=False)

    x_data_smooth = laplace_gen.from_smooth(1000, field="real")
    x_data_random = laplace_gen.from_random(1000, field="real")

    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)

    matrix = laplace_gen.sparse_operator.todense()

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
            projected_l2_loss, ops.transpose(mg.decoder.layers[-1].weights[0])
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

    gf = laplace_gen.get_gf(name="res(sin(pi*x)*sin(pi*y))")
    gf.vec.FV().NumPy()[:] = x_pred

    ng.Draw(gf, mesh, "res(sin(pi*x)*sin(pi*y))")

    # Attempt 2: Encode and decode
    gf = laplace_gen.get_gf(name="cos(pi*x)*cos(pi*y)")
    gf.Set(ng.cos(np.pi * ng.x) * ng.cos(np.pi * ng.y))
    ng.Draw(gf, mesh, "cos(pi*x)*cos(pi*y)")

    print("Predicting...")
    x_data = np.copy(gf.vec.FV().NumPy())
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    x_data = tf.reshape(x_data, (1, *x_data.shape))

    x_pred = mg.predict(x_data)

    gf_ed = laplace_gen.get_gf(name="decoded(cos(pi*x)*cos(pi*y))")
    gf_ed.vec.FV().NumPy()[:] = x_pred[0]
    ng.Draw(gf_ed, mesh, "decoded(cos(pi*x)*cos(pi*y))")


def test_vcycle_solver():
    """
    Test the Vcycle architecture with a solver
    """
    description = "Train a Vcycle, assemble a two-level solver, and solve a problem\n"
    print(description)

    # Train a Vcycle
    tf.compat.v1.enable_eager_execution()
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.05))

    laplace_gen = LaplaceDGen(mesh, tol=1e-1, is_complex=False)

    x_data_smooth = laplace_gen.from_smooth(1000, field="real")
    x_data_random = laplace_gen.from_random(1000, field="real")

    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    # x_data = tf.transpose(x_data)

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
        # metrics=[L2ErrorMetric(space=laplace_gen.ngsolve_operator.space)],
    )

    vcycle.fit(
        x_data,
        x_data,
        epochs=200,
        batch_size=500,
        shuffle=True,
        validation_data=(x_data, x_data),
    )

    # Get linear operators
    a_fine = laplace_gen.sparse_operator

    def temp_a_coarse(x_coarse):
        x_coarse = x_coarse.reshape((1, vcycle.inner_shape))
        # x_fine = vcycle.decoder.predict(x_coarse)
        x_fine = vcycle.decoder(x_coarse).numpy().flatten()
        ax_fine = a_fine @ x_fine
        ax_fine = ax_fine.reshape((1, vcycle._input_shape[-1]))
        # return vcycle.encoder.predict(ax_fine)
        return vcycle.encoder(ax_fine).numpy().flatten()

    print(f"Shape of the fine operator: {a_fine.shape}")
    print(f"Shape of the coarse operator: {vcycle.inner_shape}")
    input("Press Enter to continue...")

    a_coarse = sp.sparse.linalg.LinearOperator(
        (vcycle.inner_shape, vcycle.inner_shape),
        matvec=temp_a_coarse,
    )

    # Get right-hand side
    gf = laplace_gen.get_gf(name="1.0")
    gf.Set(1.0)
    b_fine = gf.vec.FV().NumPy()
    ng.Draw(gf, mesh, "1.0")
    input("Press Enter to continue...")

    # Assemble a two-level solver
    def two_level_solver(x_fine):
        r_fine = b_fine - a_fine @ x_fine
        # Coarse grid correction
        r_fine = r_fine.reshape((1, vcycle._input_shape[-1]))
        r_coarse = vcycle.encoder.predict(r_fine).reshape((vcycle.inner_shape,))
        e_coarse = sp.sparse.linalg.cg(a_coarse, r_coarse)[0]
        e_coarse = e_coarse.reshape((1, vcycle.inner_shape))
        e_fine = vcycle.decoder.predict(e_coarse).reshape((vcycle._input_shape[-1],))
        x_fine += e_fine
        # Smoothing
        r_fine = b_fine - a_fine @ x_fine
        e_fine = sp.sparse.linalg.cg(a_fine, r_fine)[0]
        x_fine += e_fine
        return x_fine

    # Solve a problem
    x0 = np.zeros_like(b_fine)
    for i in range(10):
        x0 = two_level_solver(x0)
        print(f"Iteration {i + 1}")
    gf_sol = laplace_gen.get_gf(name="Laplace equation solution")
    gf_sol.vec.FV().NumPy()[:] = x0
    ng.Draw(gf_sol, mesh, "Laplace equation solution")
    input("Press Enter to continue...")


if __name__ == "__main__":
    # test_vcycle_real()
    # input("Press Enter to continue...")
    # test_vcycle_complex()
    # input("Press Enter to continue...")
    # test_mg_real()
    test_vcycle_solver()
