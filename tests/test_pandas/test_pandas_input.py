import unittest

import gurobipy as gp
import gurobipy_pandas as gppd
from sklearn import datasets

from gurobi_ml import add_predictor_constr

from ..test_sklearn.sklearn_cases import DiabetesCasesAsFrame, WageCase

VERBOSE = False


class TestPandasInput(unittest.TestCase):
    @staticmethod
    def create_input(gpm, examples, numerical_features=None):
        """Create input for predictor constraints from a dataframe input"""
        if numerical_features is None:
            numerical_features = examples.columns
        x = examples.copy()
        for feat in numerical_features:
            x.loc[:, feat] = gppd.add_vars(gpm, examples, lb=feat, ub=feat)
        return x

    @staticmethod
    def create_output(gpm, examples, out_dim=1):
        """Create output for predictor constraints from a dataframe input"""
        x = gppd.add_vars(gpm, examples, lb=-gp.GRB.INFINITY)
        return x

    @staticmethod
    def dummy_model(predictor, X, numerical_features=None, create_output=False):
        with gp.Env() as env, gp.Model(env=env) as gpm:
            input_vars = TestPandasInput.create_input(gpm, X, numerical_features)
            if create_output:
                output_vars = TestPandasInput.create_output(gpm, X, 1)
            else:
                output_vars = None
            add_predictor_constr(gpm, predictor, input_vars, output_vars)

    def test_diabetes_input(self):
        data = datasets.load_diabetes(as_frame=True)

        X = data["data"][:10]
        cases = DiabetesCasesAsFrame()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            predictor = onecase["predictor"]
            numerical_features = None
            TestPandasInput.dummy_model(predictor, X, numerical_features, False)
            TestPandasInput.dummy_model(predictor, X, numerical_features, True)

    def test_wages_input(self):
        cases = WageCase()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            predictor = onecase["predictor"]
            numerical_features = cases.numerical_features
            X = onecase["data"][:10]
            TestPandasInput.dummy_model(predictor, X, numerical_features, False)
            TestPandasInput.dummy_model(predictor, X, numerical_features, True)


if __name__ == "__main__":
    VERBOSE = True
    unittest.main(verbosity=2)
