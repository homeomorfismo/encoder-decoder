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
        self,
        units,
        kernel_regularizer=None,
        initializer="glorot_uniform",
        dtype="float32",
        trainable=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = kernel_regularizer
        self.initializer = initializer
        self._dtype = dtype
        self.trainable = trainable

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.initializer,
            trainable=True,
            name="kernel",
            dtype=self._dtype,
        )

    def call(self, x):
        if self.kernel_regularizer is not None:
            self.add_loss(self.kernel_regularizer(self.kernel))
        return ops.matmul(x, self.kernel)
