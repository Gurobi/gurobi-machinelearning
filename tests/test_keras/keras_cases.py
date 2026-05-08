"""Cases to test for Keras

We want to test networks with ReLU, Sigmoid and with or without SoftMax.

"""

from abc import ABC, abstractmethod

import tensorflow as tf
import keras

try:
    from gurobipy import nlfunc  # noqa: F401

    HAS_NL_EXPR = True
except ImportError:
    HAS_NL_EXPR = False


def layers_as_string(layers):
    if isinstance(layers, str):
        return f"_{layers}"
    return "_" + "_".join(f"{layer}" for layer in layers)


class Cases(ABC):
    """Base class to have cases for testing

    This class is used to construct and save the predictors we want to test.

    It is generic in that it doesn't specify the data used.

    Attributes
    ----------
    List of architectures we want to test
    """

    def __init__(
        self,
        dataset,
    ):
        self.dataset = dataset

        # Filled with get data if needed
        self._data = None

    def __iter__(self):
        return self.all_tested_layers.__iter__()

    @abstractmethod
    def load_data(self):
        """Define this to load data for predictors"""
        ...

    @abstractmethod
    def compile(self, layers):
        """Define this to compile the neural network"""

    @property
    def data(self):
        if self._data is None:
            self.load_data()
        return self._data

    def get_case(self, layers):
        """Build model for one predictor"""
        X, y = self.data
        predictor = self.compile(layers)
        predictor.fit(X, y)

        return predictor


class HousingCases(Cases):
    """Base class to have cases for testing regression models on diabetes set

    This is appropriate for testing a regression with a single output."""

    def __init__(self):
        self.all_tested_layers = [
            [keras.layers.Dense(16, activation="relu")],
        ]
        if HAS_NL_EXPR:
            self.all_tested_layers.append(
                [keras.layers.Dense(16, activation="sigmoid")],
            )
        super().__init__("housing")
        self.load_data()

    def load_data(self):
        (X_train, y_train), (_, _) = keras.datasets.california_housing.load_data(
            version="small"
        )
        self._data = (X_train, y_train)

    def compile(self, layers):
        nn = keras.models.Sequential(
            [keras.layers.InputLayer((8,))] + layers + [keras.layers.Dense(1)]
        )
        nn.compile(loss="mean_squared_error", optimizer="adam")
        return nn


class MNISTCases(Cases):
    """Base class to have cases for testing regression models on diabetes set

    This is appropriate for testing a regression with a single output."""

    def __init__(self):
        self.all_tested_layers = [
            [keras.layers.Dense(20, activation="relu")],
        ]

        if HAS_NL_EXPR:
            self.all_tested_layers += [
                [keras.layers.Dense(20, activation="sigmoid")],
            ]
        super().__init__("housing")
        self.load_data()

    def load_data(self):
        (X_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
        X_train = tf.reshape(tf.cast(X_train, tf.float32) / 255.0, [-1, 28 * 28])
        self._data = (X_train, y_train)

    def compile(self, layers):
        nn = keras.models.Sequential(
            [keras.layers.InputLayer((28 * 28,))] + layers + [keras.layers.Dense(10)]
        )
        nn.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        return nn
