"""
Test code
"""

import ngsolve as ng
import tensorflow as tf
import keras
from geo2d import make_unit_square
from encoder import PseudoVcycle
from data_gen import gd_convection_diffusion, complex_to_real, real_to_complex
from fes import assemble, convection_diffusion


if __name__ == "__main__":
    # keras.config.set_dtype_policy("complex64")
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.1))

    matrix = ng.CF((1.0, 0.0, 0.0, 1.0), dims=(2, 2))
    vector = ng.CF((0.0, 0.0))
    scalar = ng.CF(0.0)

    a, m, fes = convection_diffusion(
        mesh, matrix, vector, scalar, order=1, is_complex=False
    )

    x_data = gd_convection_diffusion(20, precond="local", is_complex=False)
    x_data = complex_to_real(x_data)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)

    assemble(a, m)

    vcycle = PseudoVcycle(
        x_data.shape[1:],
        num_levels=1,
        compression_factor=2.0,
        regularizer=0.5,
        dtype="float32",
    )

    vcycle.compile(optimizer="adam", loss="mse")

    vcycle.fit(
        x_data,
        x_data,
        epochs=100,
        batch_size=5,
        shuffle=True,
        validation_data=(x_data, x_data),
    )
