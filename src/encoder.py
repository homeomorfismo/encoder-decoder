"""
Module that contains the encoder-decoder model for mimicking a
V-Cycle in a Multigrid solver.
"""

import keras
from keras import layers
from keras import regularizers
from keras.datasets import mnist
from keras import ops
import numpy as np
import matplotlib.pyplot as plt

# from metrics import SymL1Regularization
from layers import LinearDense
from metrics_new import SymIdL1Regularization


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
        use_bias: bool = False,
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
        # self.compression_factor = compression_factor
        self.inner_shape = int(input_shape[-1] // compression_factor)
        self.reg_param = reg_param
        self.use_bias = use_bias
        self._dtype = dtype

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

        # x = layers.Reshape((self.lin_shape,), name="reshape")(x)
        # encoder_layers.append(x)

        for j in range(self.num_levels):
            x = LinearDense(
                self.inner_shape,
                name=f"encoder_{j}",
                # kernel_regularizer=regularizers.L1(self.reg_param),
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

        for j in range(1, self.num_levels + 1):
            x = LinearDense(
                self._input_shape[-1],
                name=f"decoder_{j}",
                # kernel_regularizer=SymIdL1Regularization(
                #     self.reg_param, self.encoder.layers[j].get_weights()[0]
                # ),
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


if __name__ == "__main__":
    # see https://blog.keras.io/building-autoencoders-in-keras.html
    ENCODING_DIM = 32
    INPUT_SHAPE = (784,)
    NUM_LEVELS = 1
    COMPRESSION_FACTOR = 24.5
    REG_PARAM = 1.0e-2
    USE_BIAS = False

    model = PseudoVcycle(
        input_shape=INPUT_SHAPE,
        num_levels=NUM_LEVELS,
        compression_factor=COMPRESSION_FACTOR,
        reg_param=REG_PARAM,
        use_bias=USE_BIAS,
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
    input("Press Enter to continue...")

    # Use matplotlib.pyplot.spy to visualize the weights of the encoder
    # and decoder models.
    for i in range(1, NUM_LEVELS + 1):
        plt.figure(figsize=(10, 10))
        plt.spy(model.encoder.layers[i].get_weights()[0], markersize=1)
        plt.title(f"Encoder Layer {i}")
        plt.show()
        input("\tPress Enter to continue...")

        plt.figure(figsize=(10, 10))
        plt.spy(model.decoder.layers[i].get_weights()[0], markersize=1)
        plt.title(f"Decoder Layer {i}")
        plt.show()
        input("\tPress Enter to continue...")

    # Compute the difference between the encoder and decoder weights.
    # Use numpy.allclose to check if the weights are equal.
    for i in range(1, NUM_LEVELS + 1):
        diff_1 = (
            ops.transpose(model.encoder.layers[i].get_weights()[0])
            - model.decoder.layers[i].get_weights()[0]
        )
        print(f"Layer {i} Difference: {np.allclose(diff_1, np.zeros(diff_1.shape))}")
        diff_2 = ops.matmul(
            model.encoder.layers[i].get_weights()[0],
            model.decoder.layers[i].get_weights()[0],
        ) - np.eye(model.encoder.layers[i].get_weights()[0].shape[0])
        print(f"Layer {i} Difference: {np.allclose(diff_2, np.zeros(diff_2.shape))}")
        input("\tPress Enter to continue...")
