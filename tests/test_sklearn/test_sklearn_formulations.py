import os
import unittest
import warnings

import gurobipy as gp
import numpy as np
from gurobipy import GurobiError
from sklearn import datasets
from sklearn.pipeline import Pipeline

from gurobi_ml import add_predictor_constr, register_predictor_constr
from gurobi_ml.exceptions import NoSolution
from gurobi_ml.sklearn import add_mlp_regressor_constr
from gurobi_ml.sklearn.pipeline import PipelineConstr

from ..fixed_formulation import FixedRegressionModel
from .sklearn_cases import CircleCase, DiabetesCases, IrisCases, MNISTCase

VERBOSE = False


class TestSklearnModel(FixedRegressionModel):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    def additional_test(self, predictor, pred_constr):
        if isinstance(predictor, Pipeline):
            self.assertIsInstance(pred_constr, PipelineConstr)
            self.assertEqual(len(predictor), len(pred_constr))
            for i in range(len(pred_constr)):
                predictor_name = type(predictor[i]).__name__
                linear = ["Lasso", "Ridge"]
                if predictor_name in linear:
                    predictor_name = "LinearRegression"
                self.assertEqual(
                    predictor_name, type(pred_constr[i]).__name__[: -len("Constr")]
                )

    def test_diabetes_sklearn(self):
        data = datasets.load_diabetes()

        X = data["data"]
        cases = DiabetesCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            self.do_one_case(onecase, X, 5, "all", float_type=np.float32)
            self.do_one_case(onecase, X, 6, "pairs", float_type=np.float32)

    def test_iris_proba(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        # Make it a simple classification
        X = X[y != 2]
        y = y[y != 2]
        cases = IrisCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            self.do_one_case(onecase, X, 5, "all", output_type="probability_1")
            self.do_one_case(onecase, X, 6, "pairs", output_type="probability_1")

    def test_iris_clf(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        # Make it a simple classification
        X = X[y != 2]
        y = y[y != 2]
        cases = IrisCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            self.do_one_case(onecase, X, 5, "all", output_type="classification")
            self.do_one_case(onecase, X, 6, "pairs", output_type="classification")

    def test_iris_pwl_args(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        # Make it a simple classification
        X = X[y != 2]
        y = y[y != 2]
        cases = IrisCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            self.do_one_case(
                onecase,
                X,
                5,
                "all",
                output_type="classification",
                pwl_attributes={"FuncPieces": 5},
            )

    def test_circle(self):
        cases = CircleCase()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            X = onecase["data"]
            self.do_one_case(onecase, X, 5, "all")
            self.do_one_case(onecase, X, 6, "pairs")


class TestMNIST(unittest.TestCase):
    """Test that various versions of ReLU work and give the same results."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(1)

    def fixed_model(self, predictor, examples):
        params = {
            "OutputFlag": 0,
        }
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            lb = np.maximum(examples - 1e-4, 0.0)
            ub = np.minimum(examples + 1e-4, 1.0)
            x = gpm.addMVar(examples.shape, lb=lb, ub=ub)

            predictor.out_activation_ = "identity"
            register_predictor_constr("MLPClassifier", add_mlp_regressor_constr)
            pred_constr = add_predictor_constr(
                gpm, predictor, x, output_type="probability"
            )

            y = pred_constr.output
            with self.assertRaises(NoSolution):
                pred_constr.get_error()
            with open(os.devnull, "w") as outnull:
                pred_constr.print_stats(file=outnull)
            try:
                gpm.optimize()
            except GurobiError as E:
                if E.errno == 10010:
                    warnings.warn(UserWarning("Limited license"))
                    self.skipTest("Model too large for limited license")
                else:
                    raise

            tol = 1e-5
            vio = gpm.MaxVio
            if vio > 1e-5:
                warnings.warn(UserWarning(f"Big solution violation {vio}"))
                warnings.warn(UserWarning(f"predictor {predictor}"))
            tol = max(tol, vio)
            tol *= np.max(np.abs(y.X))
            abserror = np.abs(pred_constr.get_error()).astype(float)
            if (abserror > tol).any():
                print(f"Error: {y.X} != {predictor.predict_proba(examples)}")

            self.assertLessEqual(np.max(abserror), tol)

    def do_one_case(self, one_case, X, n_sample):
        choice = self.rng.integers(X.shape[0], size=n_sample)
        examples = X[choice, :]
        predictor = one_case["predictor"]
        with super().subTest(regressor=predictor, exampleno=choice):
            if VERBOSE:
                print(f"Doing {predictor} with example {choice}")
            self.fixed_model(predictor, examples)

    def test_mnist(self):
        cases = MNISTCase()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            X = onecase["data"]
            self.do_one_case(onecase, X, 1)
            break
            self.do_one_case(onecase, X, 3)


if __name__ == "__main__":
    VERBOSE = True
    unittest.main(verbosity=2)
