"""
Implementation of the metrics used to evaluate the performance of the models.
"""

from keras.metrics import Metric
from keras import ops


class DecodingMetric(Metric):
    """
    Metric that computes the transpose of the decoding matrix.
    """

    def __init__(self, name="decoding_transpose", decoding_matrix=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.decoding_matrix = decoding_matrix
        self.mae = self.add_variable(
            shape=(),
            initializer="zeros",
            name="mae",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric state.
        """
        diff = y_true - y_pred
        decoded_diff = ops.matmul(diff, self.decoding_matrix)
        l2_norm = ops.norm(decoded_diff, ord=2)
        if sample_weight is not None:
            sample_weight = ops.convert_to_tensor(sample_weight)
            l2_norm = l2_norm * sample_weight
        self.mae.assign_add(l2_norm)

    def result(self):
        """
        Computes the metric value.
        """
        return self.mae

    def reset_states(self):
        """
        Resets the metric state.
        """
        self.mae.assign(0.0)
