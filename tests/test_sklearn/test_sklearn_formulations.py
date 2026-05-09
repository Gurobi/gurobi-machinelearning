# Copyright © 2023-2026 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import unittest
import warnings

import gurobipy as gp
import numpy as np
from gurobipy import GurobiError
from sklearn import datasets
from sklearn.pipeline import Pipeline

from gurobi_ml import add_predictor_constr, register_predictor_constr
from gurobi_ml.exceptions import NoSolutionError
from gurobi_ml.sklearn import add_mlp_regressor_constr
from gurobi_ml.sklearn.pipeline import PipelineConstr

from ..fixed_formulation import FixedRegressionModel
from .sklearn_cases import CircleCase, DiabetesCases, MNISTCase

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
                if predictor_name in ["Lasso", "Ridge"]:
                    predictor_name = "LinearRegression"
                if predictor_name in ["PLSRegression", "PLSCanonical"]:
                    predictor_name = "PLSRegression"
                self.assertEqual(
                    predictor_name, type(pred_constr[i]).__name__[: -len("Constr")]
                )
            self.assertLessEqual(
                np.max(pred_constr[i].get_error().astype(float)),
                np.max(pred_constr.get_error().astype(float) + 1e-10),
            )

    def test_diabetes_sklearn(self):
        data = datasets.load_diabetes()

        X = data["data"]
        cases = DiabetesCases()

        for regressor in cases:
            if isinstance(regressor, Pipeline):
                actual_reg = regressor[-1]
            else:
                actual_reg = regressor
            reg_name = type(actual_reg).__name__
            if reg_name in ["RandomForestRegressor", "GradientBoostingRegressor"]:
                formulations = ["leaf"]
            else:
                formulations = [None]

            onecase = cases.get_case(regressor)
            for formulation in formulations:
                kwargs = {"float_type": np.float32, "epsilon": 1e-5}
                if formulation:
                    kwargs["formulation"] = formulation

                self.do_one_case(onecase, X, 5, "all", **kwargs)
                self.do_one_case(onecase, X, 6, "pairs", **kwargs)

                kwargs["no_debug"] = True
                self.do_one_case(onecase, X, 5, "all", **kwargs)
                self.do_one_case(onecase, X, 6, "pairs", **kwargs)

    def test_circle(self):
        cases = CircleCase()

        for regressor in cases:
            if isinstance(regressor, Pipeline):
                actual_reg = regressor[-1]
            else:
                actual_reg = regressor
            reg_name = type(actual_reg).__name__
            if reg_name in ["RandomForestRegressor", "DecisionTreeRegressor"]:
                if reg_name == "DecisionTreeRegressor":
                    formulations = ["leafs", "paths"]
                else:
                    formulations = ["leaf"]
            else:
                formulations = [None]

            onecase = cases.get_case(regressor)
            X = onecase["data"]
            for formulation in formulations:
                kwargs = {}
                if formulation:
                    kwargs["formulation"] = formulation

                self.do_one_case(onecase, X, 5, "all", **kwargs)
                self.do_one_case(onecase, X, 6, "pairs", **kwargs)

                kwargs["no_debug"] = True
                self.do_one_case(onecase, X, 5, "all", **kwargs)
                self.do_one_case(onecase, X, 6, "pairs", **kwargs)


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
            with self.assertRaises(NoSolutionError):
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
