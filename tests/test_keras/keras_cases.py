""" Cases to test for Keras

We want to test networks with ReLU, Sigmoid and with or without SoftMax.

"""

import os
from abc import ABC, abstractmethod

import keras


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
        self.basedir = os.path.join(os.path.dirname(__file__), "..", "predictors")
        self.dataset = dataset

        # Filled with get data if needed
        self._data = None

        keras_version_file = f"{dataset}_keras_version"

        try:
            with open(os.path.join(self.basedir, keras_version_file)) as file_in:
                version = file_in.read().strip()
        except FileNotFoundError:
            version = None
        if version != keras.__version__:
            print(f"Keras version changed. Regenerate predictors for {dataset}")
            self.build_all_predictors()
            with open(os.path.join(self.basedir, keras_version_file), "w") as file_out:
                print(keras.__version__, file=file_out)

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

    def predictor_file(self, predictor):
        return f"{self.dataset}_{layers_as_string(predictor)}.keras"

    def build_predictor(self, layers):
        """Build model for one predictor"""
        X, y = self.data
        predictor = self.compile(layers)
        predictor.fit(X, y)

        predictor.save(self.predictor_file(layers))
        return predictor

    def build_all_predictors(self):
        """Build all the predictor for this case.
        (Done when we have a new sklearn version)"""
        for predictor in self:
            self.build_predictor(predictor)

    def get_case(self, predictor):
        filename = self.predictor_file(predictor)
        try:
            return keras.saving.load_model(os.path.join(self.basedir, filename))
        except ValueError:
            return self.build_predictor(predictor)


class HousingCases(Cases):
    """Base class to have cases for testing regression models on diabetes set

    This is appropriate for testing a regression with a single output."""

    def __init__(self):
        self.all_tested_layers = [[keras.layers.Dense(16, activation="relu")]]
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
