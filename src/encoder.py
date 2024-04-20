"""
Module that contains the encoder-decoder model for mimicking a
V-Cycle in a Multigrid solver.
"""

import keras
from keras import layers


class PseudoVcycle(keras.Model):
    """
    Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
    """

    def __init__(
        self,
        input_shape: tuple,
        num_levels: int = 1,
        compression_factor: float = 2.0,
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

        self.compression_factor = compression_factor

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        """
        Build the encoder part of the model.

        Returns:
            keras.Model: The encoder model.
        """
        lin_shape = self._input_shape[0] * self._input_shape[1]  # Matrix shape
        final_encoding_dim = int(lin_shape / self.compression_factor)

        inputs = keras.Input(shape=(lin_shape,))
        x = inputs

        encoder_layers = []

        # TODO
        for _ in range(self.num_levels):
            x = layers.Dense(final_encoding_dim, activation="relu")(x)
            encoder_layers.append(x)
            # x = layers.MaxPooling2D()(x)

        return keras.Model(inputs, encoder_layers, name="encoder")

    def build_decoder(self):
        """
        Build the decoder part of the model.

        Returns:
            keras.Model: The decoder model.
        """
        lin_shape = self._input_shape[0] * self._input_shape[1]  # Matrix shape
        final_encoding_dim = int(lin_shape / self.compression_factor)

        inputs = keras.Input(shape=(final_encoding_dim,))
        x = inputs

        decoder_layers = []

        # TODO
        for _ in range(self.num_levels):
            x = layers.Dense(lin_shape, activation="relu")(x)
            decoder_layers.append(x)
            # x = layers.UpSampling2D()(x)

        return keras.Model(inputs, decoder_layers, name="decoder")

    def call(self, inputs, training=None, mask=None):
        """
        Call the model.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool): Whether the model is training.
            mask (tf.Tensor): Mask tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        x = self.encoder(inputs)
        x = self.decoder(x)

        return x


if __name__ == "__main__":
    model = PseudoVcycle(input_shape=(32, 32))
    model.build((None, 32 * 32))
    print(model.summary())
