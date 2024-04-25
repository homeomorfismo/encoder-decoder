"""
Initializers for the model.
"""

from keras.initializers import Initializer


class MatrixInitializer(Initializer):
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, shape, dtype=None):
        return self.matrix

    def get_config(self):
        return {"matrix": self.matrix}
