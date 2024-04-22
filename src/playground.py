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
    mesh = ng.Mesh(make_unit_square().GenerateMesh(maxh=0.1))

    laplace_dgen = LaplaceDGen(mesh, precond="local", is_complex=True)

    x_data_smooth = laplace_dgen.from_smooth(50)
    x_data_random = laplace_dgen.from_random(50)
    x_data = np.concatenate((x_data_smooth, x_data_random), axis=0)
    x_data = complex_to_real(x_data)
    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)

    vcycle = PseudoVcycle(
        x_data.shape[1:],
        num_levels=1,
        compression_factor=2.0,
        regularizer=0.001,
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

    x_data_smooth = laplace_dgen.from_smooth(1)
    x_predict = complex_to_real(x_data_smooth)
    x_predict = tf.convert_to_tensor(x_predict, dtype=tf.float32)
    x_predict = vcycle.predict(x_predict)
    type(x_predict)
    print(x_predict)
    # x_predict = real_to_complex(x_predict.numpy())
    # np.savetxt("x_predict.txt", x_predict)
    # np.savetxt("x_data_smooth.txt", x_data_smooth)
