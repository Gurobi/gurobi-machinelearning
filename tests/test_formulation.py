import os
import unittest
import warnings

import base_cases
import gurobipy as gp
import numpy as np
import tensorflow as tf
import torch
from gurobipy import GurobiError
from sklearn import datasets
from sklearn.pipeline import Pipeline

from gurobi_ml import add_predictor_constr, register_predictor_constr
from gurobi_ml.exceptions import NoSolution
from gurobi_ml.sklearn import add_mlp_regressor_constr
from gurobi_ml.sklearn.pipeline import PipelineConstr

VERBOSE = False


class TestFixedRegressionModel(unittest.TestCase):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    def fixed_model(self, predictor, examples, nonconvex, **kwargs):
        params = {
            "OutputFlag": 0,
            "NonConvex": 2,
        }
        for param in params:
            try:
                params[param] = int(params[param])
            except ValueError:
                pass

        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar(examples.shape, lb=examples - 1e-4, ub=examples + 1e-4)

            pred_constr = add_predictor_constr(gpm, predictor, x, **kwargs)

            y = pred_constr.output

            if isinstance(predictor, Pipeline):
                self.assertIsInstance(pred_constr, PipelineConstr)
                self.assertEqual(len(predictor), len(pred_constr))
                for i in range(len(pred_constr)):
                    predictor_name = type(predictor[i]).__name__
                    linear = ["Lasso", "Ridge"]
                    if predictor_name in linear:
                        predictor_name = "LinearRegression"
                    self.assertEqual(predictor_name, type(pred_constr[i]).__name__[: -len("Constr")])
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

            if nonconvex:
                tol = 5e-3
            else:
                tol = 1e-5
            vio = gpm.MaxVio
            if vio > 1e-5:
                warnings.warn(UserWarning(f"Big solution violation {vio}"))
                warnings.warn(UserWarning(f"predictor {predictor}"))
            tol = max(tol, vio)
            tol *= np.max(np.abs(y.X))
            abserror = np.abs(pred_constr.get_error()).astype(float)
            if (abserror > tol).any():
                print(f"Error: {y.X} != {predictor.predict(examples)}")

            self.assertLessEqual(np.max(abserror), tol)

    def all_of_them(self):  # pragma: no cover
        data = datasets.load_diabetes()

        X = data["data"]
        cases = base_cases.DiabetesCases()

        onecase = cases.get_case(cases.all_test[17])
        regressor = onecase["predictor"]
        for exampleno in range(X.shape[0]):
            with super().subTest(regressor=regressor, exampleno=exampleno):
                print("Test {}".format(exampleno))
                self.fixed_model(regressor, X[exampleno, :].astype(np.float32), onecase["nonconvex"])

    def do_one_case(self, one_case, X, n_sample, combine, **kwargs):
        choice = np.random.randint(X.shape[0], size=n_sample)
        examples = X[choice, :]
        if combine == "all":
            # Do the average case
            examples = (examples.sum(axis=0) / n_sample).reshape(1, -1)
        elif combine == "pairs":
            np.random.shuffle(examples)
            # Make pairwise combination of the examples
            even_rows = examples[::2, :]
            odd_rows = examples[1::2, :]
            assert odd_rows.shape == even_rows.shape
            examples = (even_rows + odd_rows) / 2.0 - 1e-2
            assert examples.shape == even_rows.shape

        predictor = one_case["predictor"]
        with super().subTest(regressor=predictor, exampleno=choice, n_sample=n_sample, combine=combine):
            if VERBOSE:
                print(f"Doing {predictor} with example {choice}")
            self.fixed_model(predictor, examples, one_case["nonconvex"], **kwargs)

    def test_diabetes_sklearn(self):
        data = datasets.load_diabetes()

        X = data["data"]
        cases = base_cases.DiabetesCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            self.do_one_case(onecase, X, 5, "all", float_type=np.float32)
            self.do_one_case(onecase, X, 6, "pairs", float_type=np.float32)

    def test_diabetes_pytorch(self):
        data = datasets.load_diabetes()

        X = data["data"]

        filename = os.path.join(os.path.dirname(__file__), "predictors", "diabetes__pytorch.pt")
        regressor = torch.load(filename)
        onecase = {"predictor": regressor, "nonconvex": 0}
        self.do_one_case(onecase, X, 5, "all")
        self.do_one_case(onecase, X, 6, "pairs")

    def test_diabetes_keras(self):
        data = datasets.load_diabetes()

        X = data["data"]

        filename = os.path.join(os.path.dirname(__file__), "predictors", "diabetes_keras")
        regressor = tf.keras.models.load_model(filename)
        onecase = {"predictor": regressor, "nonconvex": 0}
        self.do_one_case(onecase, X, 5, "all")
        self.do_one_case(onecase, X, 6, "pairs")

    def test_diabetes_keras_alt(self):
        data = datasets.load_diabetes()

        X = data["data"]

        filename = os.path.join(os.path.dirname(__file__), "predictors", "diabetes_keras_v2")
        regressor = tf.keras.models.load_model(filename)
        onecase = {"predictor": regressor, "nonconvex": 0}
        self.do_one_case(onecase, X, 5, "all")
        self.do_one_case(onecase, X, 6, "pairs")

    def test_iris_proba(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        # Make it a simple classification
        X = X[y != 2]
        y = y[y != 2]
        cases = base_cases.IrisCases()

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
        cases = base_cases.IrisCases()

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
        cases = base_cases.IrisCases()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            self.do_one_case(onecase, X, 5, "all", output_type="classification", pwl_attributes={"FuncPieces": 5})

    def test_circle(self):
        cases = base_cases.CircleCase()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            X = onecase["data"]
            self.do_one_case(onecase, X, 5, "all")
            self.do_one_case(onecase, X, 6, "pairs")


class TestMNIST(unittest.TestCase):
    """Test that various versions of ReLU work and give the same results."""

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
            pred_constr = add_predictor_constr(gpm, predictor, x, output_type="probability")

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
        choice = np.random.randint(X.shape[0], size=n_sample)
        examples = X[choice, :]
        predictor = one_case["predictor"]
        with super().subTest(regressor=predictor, exampleno=choice):
            if VERBOSE:
                print(f"Doing {predictor} with example {choice}")
            self.fixed_model(predictor, examples)

    def test_mnist(self):
        cases = base_cases.MNISTCase()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            X = onecase["data"]
            self.do_one_case(onecase, X, 1)
            break
            self.do_one_case(onecase, X, 3)


if __name__ == "__main__":
    VERBOSE = True
    unittest.main(verbosity=2)
