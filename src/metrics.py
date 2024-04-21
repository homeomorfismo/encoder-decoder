"""
Metrics for evaluating the performance of the PseudoVcycle model/encoder.
"""

from keras.metrics import Metric
from keras import ops
from keras import regularizers


class SymL1Regularization(regularizers.Regularizer):
    """
    L1 regularization for symmetric matrices.
    Given an input matrix V, the symmetric L1 regularization term is the sum of
        ||W - V^T||_1 + ||W||_1
    """

    def __init__(self, strength: float, weight_matrix):
        """
        Initialize the regularizer.

        Args:
            strength (float): The regularization strength.
            V (tf.Tensor): The matrix V.
        """
        self.strength = strength
        self.transpose = ops.transpose(weight_matrix)

    def __call__(self, W):
        """
        Compute the regularization term.

        Args:
            W (tf.Tensor): The matrix W.

        Returns:
            tf.Tensor: The regularization term.
        """
        return self.strength * ops.norm(
            W - self.transpose, ord=1
        ) + self.strength * ops.norm(W, ord=1)

    def get_config(self):
        """
        Get the configuration of the regularizer.

        Returns:
            dict: The configuration of the regularizer.
        """
        return {"strength": self.strength, "transpose": self.transpose}


class MseL1Regularization(Metric):
    """
    Mean Squared Error with L1 regularization metric for the PseudoVcycle model.
    The L1 regularization term applies to the weights of the model. This is,
    if the encoder has a weight matrix W and the decoder has a weight matrix V,
    then the L1 regularization term is the sum of the absolute values of the
    elements of W and V:
        ||W||_1 + ||V||_1 + ||W - V^T||_1 + MSE
    """

    def __init__(self, name="mse_l1_regularization", **kwargs):
        """
        Initialize the metric.

        Args:
            name (str): Name of the metric.

        Members:
            mse (tf.Tensor): Mean Squared Error.
            l1_norms (tf.Tensor): L1 norms of the weights.
        """
        super().__init__(name=name, **kwargs)
        self.mse = self.add_weight(name="mse", initializer="zeros")
        self.l1_norms = self.add_weight(name="l1_norms", initializer="zeros")

    def update_state(
        self, y_true, y_pred, sample_weight=None, encoder=None, decoder=None
    ):
        """
        Update the metric state.

        Args:
            y_true (tf.Tensor): True values.
            y_pred (tf.Tensor): Predicted values.
            sample_weight (tf.Tensor): Sample weights.
        """
        assert encoder is not None, "Encoder model must be provided."
        assert decoder is not None, "Decoder model must be provided."

        self.mse.assign_add(ops.norm(y_true - y_pred) ** 2)

        self.l1_norms.assign_add(ops.norm(ops.reshape(encoder.weights[0], [-1]), ord=1))
        self.l1_norms.assign_add(ops.norm(encoder.weights[1], ord=1))
        self.l1_norms.assign_add(ops.norm(ops.reshape(decoder.weights[0], [-1]), ord=1))
        self.l1_norms.assign_add(ops.norm(decoder.weights[1], ord=1))
        self.l1_norms.assign_add(
            ops.norm(encoder.weights[0] - ops.transpose(decoder.weights[0]), ord=1)
        )
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            self.l1_norms.assign(ops.multiply(self.l1_norms, sample_weight))

    def result(self):
        """
        Compute the metric result.
        """
        return self.mse + self.l1_norms

    def reset_states(self):
        """
        Reset the metric state.
        """
        self.mse.assign(0.0)
        self.l1_norms.assign(0.0)


if __name__ == "__main__":
    import numpy as np
    import keras
    from keras import layers

    # From Keras encoder-decoder example
    encoding_dim = 2
    input_img = keras.Input(shape=(4,))
    encoded = layers.Dense(encoding_dim, activation="relu")(input_img)
    decoded = layers.Dense(4, activation="sigmoid")(encoded)

    autoencoder = keras.Model(input_img, decoded)

    # Create encoder and decoder models
    encoder = keras.Model(input_img, encoded)
    encoded_input = keras.Input(shape=(encoding_dim,))

    decoder_layer = autoencoder.layers[-1]
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    # Create the metric
    metric = MseL1Regularization()

    # Update the metric
    y_true = np.random.rand(10, 4)
    y_pred = np.random.rand(10, 4)
    metric.update_state(y_true, y_pred, encoder=encoder, decoder=decoder)

    # Compute the metric result
    result = metric.result()
    print(result)
    assert result.shape == ()

    # Reset the metric
    metric.reset_states()
    result = metric.result()
    print(result)
    assert result == 0.0
