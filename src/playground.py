"""
Playground for testing the encoder-decoder architecture
"""

from functools import partial
import ngsolve as ng
import numpy as np
import tensorflow as tf

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


# def test_mg(is_complex=True):
# MGLike
# Requires the matrix as a tensor: scipy CSR to sparse tensor
# matrix = tf.convert_to_tensor(
#         laplace_gen.sparse_operator,
#         dtype=tf.float32
#     )
# mg = PseudoMG(
#     x_data.shape[1:],
#     matrix=matrix,
#     num_levels=1,
#     compression_factor=5.0,
#     reg_param=1e-5,
#     dtype="float32",
# )

# mg.compile(
#     optimizer="adam",
#     loss=partial(projected_l2_loss, matrix),
#     # metrics=[L2ErrorMetric(space=laplace_gen.ngsolve_operator.space)],
# )


if __name__ == "__main__":
    test_vcycle_real()
    input("Press Enter to continue...")
    test_vcycle_complex()
    input("Press Enter to continue...")
    # test_mg()
