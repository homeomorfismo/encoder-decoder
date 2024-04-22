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

        # TODO successive compression factors!
        self.num_levels = num_levels
        self.compression_factor = compression_factor
        self._input_shape = input_shape

        self.lin_shape = 1
        if len(self._input_shape) > 1:
            for size in self._input_shape:
                self.lin_shape *= size
        elif len(self._input_shape) == 1:
            self.lin_shape = self._input_shape[0]
        else:
            raise ValueError(f"Invalid input shape: {self._input_shape}")

        # TODO Generate a list of encoding dimensions!
        self.encoding_dim = int(self.lin_shape // self.compression_factor)

        self.regularizer = regularizer
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

        for i in range(self.num_levels):
            x = layers.Dense(
                self.encoding_dim,
                activation="relu",
                kernel_regularizer=regularizers.l1(self.regularizer),
                use_bias=self.use_bias,
                bias_regularizer=regularizers.l1(self.regularizer),
                name=f"encoder_{i}",
                dtype=self._dtype,
            )(x)
            encoder_layers.append(x)

        return keras.Model(inputs, encoder_layers, name="encoder")

    def build_decoder(self):
        """
        Build the decoder part of the model.

        Returns:
            keras.Model: The decoder model.
        """
        inputs = keras.Input(shape=(self.encoding_dim,))
        x = inputs

        decoder_layers = []

        for i in range(self.num_levels):
            x = layers.Dense(
                self.lin_shape,
                activation="relu",
                kernel_regularizer=SymL1Regularization(
                    # self.regularizer, self.encoder.layers[i + 2].get_weights()[0]
                    self.regularizer,
                    self.encoder.layers[i + 1].get_weights()[0],
                ),
                use_bias=self.use_bias,
                bias_regularizer=regularizers.l1(self.regularizer),
                name=f"decoder_{i}",
                dtype=self._dtype,
            )(x)
            decoder_layers.append(x)

        # x = layers.Reshape(self._input_shape, name="reshape")(x)
        # decoder_layers.append(x)

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
