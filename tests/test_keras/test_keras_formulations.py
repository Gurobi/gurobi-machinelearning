import os

import keras
from joblib import load

from ..fixed_formulation import FixedRegressionModel

VERBOSE = False


class TestKerasModel(FixedRegressionModel):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    basedir = os.path.join(os.path.dirname(__file__), "..", "predictors")

    def test_diabetes_keras(self):
        X = load(os.path.join(self.basedir, "examples_diabetes.joblib"))

        filename = os.path.join(self.basedir, "diabetes.keras")
        regressor = keras.saving.load_model(filename)
        onecase = {"predictor": regressor, "nonconvex": 0}
        self.do_one_case(onecase, X, 5, "all")
        self.do_one_case(onecase, X, 6, "pairs")

    def test_conv2d_layers(self):
        rng = self.rng
        model = keras.Sequential(
            [
                keras.layers.InputLayer((4, 4, 1)),
                keras.layers.Conv2D(2, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1),
            ]
        )

        for layer in model.layers:
            if hasattr(layer, "kernel_initializer"):
                weights = [rng.random(w.shape) for w in layer.get_weights()]
                layer.set_weights(weights)

        X = rng.random((4, 4, 4, 1))
        onecase = {"predictor": model, "nonconvex": 0}
        self.do_one_case(onecase, X, 3, "all")

    def test_diabetes_keras_alt(self):
        X = load(os.path.join(self.basedir, "examples_diabetes.joblib"))

        filename = os.path.join(
            os.path.dirname(__file__), "..", "predictors", "diabetes_v2.keras"
        )
        print(filename)
        regressor = keras.saving.load_model(filename)
        onecase = {"predictor": regressor, "nonconvex": 0}
        self.do_one_case(onecase, X, 5, "all")
        self.do_one_case(onecase, X, 6, "pairs")
