"""
Implementation of the layers used in the model.
"""

from keras.layers import Layer
from keras import ops


class LinearDense(Layer):
    """
    Custom layer for a linear dense layer.
    """

    def __init__(
        self, units, kernel_regularizer=None, initializer="glorot_uniform", **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = kernel_regularizer
        self.initializer = initializer

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.initializer,
            trainable=True,
            name="kernel",
        )

    def call(self, x):
        if self.kernel_regularizer is not None:
            self.add_loss(self.kernel_regularizer(self.kernel))
        return ops.matmul(x, self.kernel)
