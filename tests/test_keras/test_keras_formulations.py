import os

import keras
from joblib import load

try:
    from gurobipy import nlfunc  # noqa: F401

    HAS_NL_EXPR = True
except ImportError:
    HAS_NL_EXPR = False

from ..fixed_formulation import FixedRegressionModel
from .keras_cases import HousingCases, MNISTCases

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

    def test_housing_keras(self):
        cases = HousingCases()

        X = cases._data[0]
        for case in cases:
            regressor = cases.get_case(case)
            onecase = {"predictor": regressor, "nonconvex": 0}
            self.do_one_case(onecase, X, 5, "all")
            self.do_one_case(onecase, X, 6, "pairs")

    def test_mnist_keras(self):
        cases = MNISTCases()

        X = cases._data[0].numpy()
        for case in cases:
            regressor = cases.get_case(case)
            onecase = {"predictor": regressor, "nonconvex": 0}
            self.do_one_case(onecase, X, 5, "all")
            self.do_one_case(onecase, X, 6, "pairs")
