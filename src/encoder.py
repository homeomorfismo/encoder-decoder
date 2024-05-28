"""
Module that contains the encoder-decoder model for mimicking a
V-Cycle in a Multigrid solver.
"""

import keras
from keras import regularizers
from keras.datasets import mnist
from keras import ops
import numpy as np
import matplotlib.pyplot as plt

from layers import LinearDense
from regularizers import SymIdL1Regularization, IdRegularization
from initializers import MatrixInitializer


class PseudoVcycle(keras.Model):
    """
    Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
    """

    def __init__(
        self,
        input_shape: tuple,
        num_levels: int = 1,
        compression_factor: float = 2.0,
        reg_param: float = 1.0e-4,
        initializer_encoder="glorot_uniform",
        initializer_decoder="zeros",
        dtype="float32",
    ):
        """
        Constructor for the PseudoVcycle model.

        Args:
            input_shape (tuple): Shape of the input tensor.
            num_levels (int): Number of levels in the V-Cycle.
        """
        super().__init__()

        self._name = "PseudoVcycle"

        self._input_shape = input_shape
        self.num_levels = num_levels
        self.inner_shape = int(input_shape[-1] // compression_factor)
        # self.inner_shapes = [int(input_shape[-1] // (compression_factor ** j)) for j in range(1, num_levels + 1)]
        self.reg_param = reg_param
        self._dtype = dtype

        self.initializer_encoder = initializer_encoder
        self.initializer_decoder = initializer_decoder

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    def build_encoder(self):
        """
        Build the encoder part of the model.

        Returns:
            keras.Model: The encoder model.
        """
        inputs = keras.Input(shape=self._input_shape)
        x = inputs

        encoder_layers = []

        for j in range(self.num_levels):
            x = LinearDense(
                self.inner_shape,
                # self.inner_shapes[j],
                name=f"encoder_{j}",
                kernel_regularizer=IdRegularization(
                    self.reg_param, self.inner_shape, transpose=True
                ),
                initializer=self.initializer_encoder,
                dtype=self.dtype,
            )(x)
            encoder_layers.append(x)

        return keras.Model(inputs, encoder_layers, name="encoder")

    def build_decoder(self):
        """
        Build the decoder part of the model.

        Returns:
            keras.Model: The decoder model.
        """
        inputs = keras.Input(shape=(self.inner_shape,))
        # inputs = keras.Input(shape=(self.inner_shapes[-1],))
        x = inputs

        decoder_layers = []

        for j in range(self.num_levels):
            x = LinearDense(
                self._input_shape[-1],
                # self.inner_shapes[-j+1],
                name=f"decoder_{j}",
                # kernel_regularizer=IdRegularization(
                #     self.reg_param, self.inner_shape, transpose=False
                # ),
                kernel_regularizer=SymIdL1Regularization(
                    self.reg_param,
                    self.encoder.get_layer(f"encoder_{self.num_levels - j - 1}").kernel,
                ),
                initializer=self.initializer_decoder,
                dtype=self.dtype,
            )(x)
            decoder_layers.append(x)

        return keras.Model(inputs, decoder_layers, name="decoder")

    def call(self, inputs, training=None):
        """
        Call the model.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool): Whether the model is training.
            mask (tf.Tensor): Mask tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        x = self.encoder(inputs, training=training)
        x = self.decoder(x, training=training)
        return x


class PseudoMG(keras.Model):
    """
    Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
    """

    def __init__(
        self,
        input_shape: tuple,
        matrix=None,
        num_levels: int = 1,
        compression_factor: float = 2.0,
        reg_param: float = 1.0e-4,
        initializer_encoder="glorot_uniform",
        initializer_decoder="zeros",
        dtype="float32",
    ):
        """
        Constructor fo the PseudoMG model.
        """
        assert matrix is not None, "Matrix must be provided."
        super().__init__()

        self._name = "PseudoMG"
        self.matrix = matrix

        self._input_shape = input_shape
        self.num_levels = num_levels
        # self.inner_shapes = [int(input_shape[-1] // (compression_factor ** j)) for j in range(1, num_levels + 1)]
        self.inner_shape = int(input_shape[-1] // compression_factor)
        self.reg_param = reg_param
        self._dtype = dtype

        self.initializer_encoder = initializer_encoder
        self.initializer_decoder = initializer_decoder

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.range_space = self.build_range_space(self.matrix)

        self.range_space.trainable = False

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    def build_encoder(self):
        """
        Build the encoder part of the model.

        Returns:
            keras.Model: The encoder model.
        """
        inputs = keras.Input(shape=self._input_shape)
        x = inputs

        encoder_layers = []

        for j in range(self.num_levels):
            x = LinearDense(
                self.inner_shape,
                name=f"encoder_{j}",
                kernel_regularizer=IdRegularization(
                    self.reg_param, self.inner_shape, transpose=True
                ),
                initializer=self.initializer_encoder,
                dtype=self.dtype,
            )(x)
            encoder_layers.append(x)

        return keras.Model(inputs, encoder_layers, name="encoder")

    def build_decoder(self):
        """
        Build the decoder part of the model.

        Returns:
            keras.Model: The decoder model.
        """
        inputs = keras.Input(shape=(self.inner_shape,))
        x = inputs

        decoder_layers = []

        for j in range(self.num_levels):
            x = LinearDense(
                self._input_shape[-1],
                name=f"decoder_{j}",
                # kernel_regularizer=IdRegularization(
                #     self.reg_param, self.inner_shape, transpose=False
                # ),
                kernel_regularizer=SymIdL1Regularization(
                    self.reg_param,
                    self.encoder.get_layer(f"encoder_{self.num_levels - j - 1}").kernel,
                ),
                initializer=self.initializer_decoder,
                dtype=self.dtype,
            )(x)
            decoder_layers.append(x)

        return keras.Model(inputs, decoder_layers, name="decoder")

    def build_range_space(self, matrix):
        """
        Build the range space of the model.

        Args:
            matrix (tf.Tensor): Matrix tensor.

        Returns:
            keras.Model: The range space model.
        """
        inputs = keras.Input(shape=self._input_shape)
        x = inputs

        range_space_layers = []

        x = LinearDense(
            self.inner_shape,
            name="range_space_0",
            initializer=MatrixInitializer(matrix),
            dtype=self.dtype,
            trainable=False,
        )(x)
        range_space_layers.append(x)

        return keras.Model(inputs, range_space_layers, name="range_space")

    def call(self, inputs, training=None):
        """
        Call the model.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool): Whether the model is training.
            mask (tf.Tensor): Mask tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        x = self.encoder(inputs, training=training)
        x = self.decoder(x, training=training)
        x = self.range_space(x, training=False)
        return x


# see https://blog.keras.io/building-autoencoders-in-keras.html
ENCODING_DIM = 32
INPUT_SHAPE = (784,)
NUM_LEVELS = 1
COMPRESSION_FACTOR = 24.5
REG_PARAM = 1.0e-2
DTYPE = "float32"


def test_pseudo_vcycle():
    """
    Test the PseudoVcycle model. Use the MNIST dataset.
    """
    model = PseudoVcycle(
        input_shape=INPUT_SHAPE,
        num_levels=NUM_LEVELS,
        compression_factor=COMPRESSION_FACTOR,
        reg_param=REG_PARAM,
        dtype=DTYPE,
    )
    model.compile(optimizer="adam", loss="mean_absolute_error")

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_data=(x_test, x_test),
    )

    encoded_imgs = model.encoder.predict(x_test)
    decoded_imgs = model.decoder.predict(encoded_imgs)

    N = 10
    plt.figure(figsize=(20, 4))
    for i in range(N):
        # display original
        ax = plt.subplot(2, N, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, N, i + 1 + N)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def test_pseudo_mg():
    """
    Test the PseudoMG model. Use the MNIST dataset.
    """
    model = PseudoMG(
        input_shape=INPUT_SHAPE,
        matrix=ops.eye(784),
        num_levels=NUM_LEVELS,
        compression_factor=COMPRESSION_FACTOR,
        reg_param=REG_PARAM,
        dtype=DTYPE,
    )
    model.compile(optimizer="adam", loss="mean_absolute_error")

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_data=(x_test, x_test),
    )

    encoded_imgs = model.encoder.predict(x_test)
    decoded_imgs = model.decoder.predict(encoded_imgs)

    N = 10
    plt.figure(figsize=(20, 4))
    for i in range(N):
        # display original
        ax = plt.subplot(2, N, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, N, i + 1 + N)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    test_pseudo_vcycle()
    test_pseudo_mg()
