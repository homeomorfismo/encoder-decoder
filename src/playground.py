"""
Test code
"""

import ngsolve as ng
import numpy as np
import tensorflow as tf
from geo2d import make_unit_square
from encoder import PseudoVcycle
from data_gen import real_to_complex, complex_to_real, LaplaceDGen

# from fes import assemble, convection_diffusion


if __name__ == "__main__":
    # keras.config.set_dtype_policy("complex64")
    tf.compat.v1.enable_eager_execution()
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.05))

    laplace_dgen = LaplaceDGen(mesh, precond="local", is_complex=True)

    x_data_smooth = laplace_dgen.from_smooth(2000)
    x_data_random = laplace_dgen.from_random(2000)
    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = complex_to_real(x_data)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)

    vcycle = PseudoVcycle(
        x_data.shape[1:],
        num_levels=1,
        compression_factor=20.0,
        regularizer=1.0e-6,
        dtype="float32",
    )

    vcycle.compile(optimizer="adam", loss="mean_absolute_error")

    vcycle.fit(
        x_data,
        x_data,
        epochs=200,
        batch_size=100,
        shuffle=True,
        validation_data=(x_data, x_data),
    )

    x_data_smooth = laplace_dgen.from_smooth(1)
    print(type(x_data_smooth))
    print(x_data_smooth)
    input("Press Enter to continue...")

    x_predict = complex_to_real(x_data_smooth)
    x_predict = tf.convert_to_tensor(x_predict, dtype=tf.float32)
    x_predict = vcycle.predict(x_predict)
    print(type(x_predict))
    print(x_predict)
