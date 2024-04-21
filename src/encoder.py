"""
Module that contains the encoder-decoder model for mimicking a
V-Cycle in a Multigrid solver.
"""

import keras
from keras import layers
from keras import regularizers
from metrics import SymL1Regularization


class PseudoVcycle(keras.Model):
    """
    Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
    """

    def __init__(
        self,
        input_shape: tuple,
        num_levels: int = 1,
        compression_factor: float = 2.0,
        regularizer: float = 1.0,
    ):
        """
        Constructor for the PseudoVcycle model.

        Args:
            input_shape (tuple): Shape of the input tensor.
            num_levels (int): Number of levels in the V-Cycle.
        """
        super().__init__()

        self._name = "PseudoVcycle"

        self.num_levels = num_levels
        self._input_shape = input_shape
        # self._output_shape = input_shape

        self.compression_factor = compression_factor
        self.regularizer = regularizer

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        """
        Build the encoder part of the model.

        Returns:
            keras.Model: The encoder model.
        """
        if len(self._input_shape) == 2:
            lin_shape = self._input_shape[0] * self._input_shape[1]  # Matrix shape
        else:
            lin_shape = self._input_shape[0]
        final_encoding_dim = int(lin_shape / self.compression_factor)

        # inputs = keras.Input(shape=(lin_shape,))
        inputs = keras.Input(shape=self._input_shape)
        x = inputs

        encoder_layers = []

        # Flatten the input tensor
        x = layers.Reshape((lin_shape,), name="reshape")(x)
        encoder_layers.append(x)

        for i in range(self.num_levels):
            x = layers.Dense(
                final_encoding_dim,
                activation="relu",
                kernel_regularizer=regularizers.l1(self.regularizer),
                bias_regularizer=regularizers.l1(self.regularizer),
                name=f"encoder_{i}",
            )(x)
            encoder_layers.append(x)

        return keras.Model(inputs, encoder_layers, name="encoder")

    def build_decoder(self):
        """
        Build the decoder part of the model.

        Returns:
            keras.Model: The decoder model.
        """
        if len(self._input_shape) == 2:
            lin_shape = self._input_shape[0] * self._input_shape[1]  # Matrix shape
        else:
            lin_shape = self._input_shape[0]
        final_encoding_dim = int(lin_shape / self.compression_factor)

        inputs = keras.Input(shape=(final_encoding_dim,))
        x = inputs

        decoder_layers = []

        for i in range(self.num_levels):
            x = layers.Dense(
                lin_shape,
                activation="relu",
                kernel_regularizer=SymL1Regularization(
                    self.regularizer, self.encoder.layers[i + 2].get_weights()[0]
                ),
                bias_regularizer=regularizers.l1(self.regularizer),
                name=f"decoder_{i}",
            )(x)
            decoder_layers.append(x)

        # Reshape the output tensor
        x = layers.Reshape(self._input_shape, name="reshape")(x)
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
    model = PseudoVcycle(input_shape=(32, 32))
    model.build((None, 32 * 32))
    print(model.summary())
    print(model.encoder.summary())
    print(model.decoder.summary())

    print("\nEncoder:")
    for layer in model.encoder.layers:
        print(f"{layer.name}")

    print("\nDecoder:")
    for layer in model.decoder.layers:
        print(f"{layer.name}")

    print("\nModel:")
    for layer in model.layers:
        print(f"{layer.name}")
