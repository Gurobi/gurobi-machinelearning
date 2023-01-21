import os
import unittest
import warnings

import gurobipy as gp
import numpy as np
from gurobipy import GurobiError

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import NoSolution

VERBOSE = False


class FixedRegressionModel(unittest.TestCase):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(1)

    def additional_test(self, predictor, pred_constr):
        """Define this to do additional tests"""

    def fixed_model(
        self, predictor, examples, nonconvex, numerical_features=None, **kwargs
    ):
        params = {
            "OutputFlag": 0,
        }
        if nonconvex:
            params["NonConvex"] = 2
        for param in params:
            try:
                params[param] = int(params[param])
            except ValueError:
                pass

        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            if numerical_features:
                import gurobipy_pandas as gppd
                import pandas as pd

                self.assertIsInstance(examples, pd.DataFrame)
                self.assertIsInstance(numerical_features, list)
                x = examples.copy()
                nonconvex = 1
                for feat in numerical_features:
                    x.loc[:, feat] = gppd.add_vars(gpm, examples, lb=feat, ub=feat)
            else:
                x = gpm.addMVar(examples.shape, lb=examples - 1e-4, ub=examples + 1e-4)
                if hasattr(examples, "columns"):
                    import pandas as pd

                    x = pd.DataFrame(data=x.tolist(), columns=examples.columns)

            gpm.update()
            pred_constr = add_predictor_constr(gpm, predictor, x, **kwargs)

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

            self.additional_test(predictor, pred_constr)
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
            abserror = pred_constr.get_error().astype(float)
            if (abserror > tol).any():
                print(f"Error: {y.X} != {predictor.predict(examples)}")

            self.assertLessEqual(np.max(abserror), tol)

    def do_one_case(self, one_case, X, n_sample, combine="", **kwargs):
        choice = self.rng.choice(X.shape[0], size=n_sample, replace=False)
        if hasattr(X, "columns"):
            examples = X.iloc[choice, :].copy()
        else:
            examples = X[choice, :]
        if combine == "all":
            # Do the average case
            if hasattr(X, "columns"):
                examples.iloc[0, :] = examples.mean()
                examples = examples.iloc[:1, :]
            else:
                examples = (examples.sum(axis=0) / n_sample).reshape(1, -1) - 1e-2
        elif combine == "pairs":
            # Make pairwise combination of the examples
            if hasattr(X, "columns"):
                even_rows = examples.iloc[::2, :] / 2.0
                odd_rows = examples.iloc[1::2, :] / 2.0
                odd_rows.index = even_rows.index
            else:
                even_rows = examples[::2, :] / 2.0
                odd_rows = examples[1::2, :] / 2.0
            assert odd_rows.shape == even_rows.shape
            examples = (even_rows + odd_rows) - 1e-2
            assert examples.shape == even_rows.shape

        predictor = one_case["predictor"]
        with super().subTest(
            regressor=predictor, exampleno=choice, n_sample=n_sample, combine=combine
        ):
            if VERBOSE:
                print(f"Doing {predictor} with example {choice}")
            self.fixed_model(predictor, examples, one_case["nonconvex"], **kwargs)
