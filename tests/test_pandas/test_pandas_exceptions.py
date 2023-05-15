import unittest
import warnings

import gurobipy as gp
import gurobipy_pandas as gppd
import numpy as np
from sklearn import datasets
from sklearn.compose import make_column_transformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import NoModel


class TestUnsuportedPandas(unittest.TestCase):
    @staticmethod
    def create_input(gpm, examples, numerical_features=None):
        """Create input for predictor constraints from a dataframe input"""
        if numerical_features is None:
            numerical_features = examples.columns
        x = examples.copy()
        for feat in numerical_features:
            x.loc[:, feat] = gppd.add_vars(gpm, examples, lb=feat, ub=feat)
        return x

    def test_pipeline_fail_transformer_idx(self):
        data = datasets.load_diabetes(as_frame=True)

        X = data.data
        y = data.target

        coltran = make_column_transformer(
            (StandardScaler(), [0, 1, 2, 3]),
            (FunctionTransformer(np.exp), [4, 5, 6, 7, 8, 9]),
        )
        mlpreg = make_pipeline(coltran, MLPRegressor([10] * 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            mlpreg.fit(X, y)
        example = X.iloc[:5, :]

        m = gp.Model()

        x = self.create_input(m, example)

        with self.assertRaises(NoModel):
            add_predictor_constr(m, mlpreg, x)

    def test_pipeline_fail_transformer_str(self):
        data = datasets.load_diabetes(as_frame=True)

        X = data.data
        y = data.target

        coltran = make_column_transformer(
            (StandardScaler(), ["age", "sex", "bmi", "bp"]),
            (FunctionTransformer(np.exp), ["s1", "s2", "s3", "s4", "s5", "s6"]),
        )
        mlpreg = make_pipeline(coltran, MLPRegressor([10] * 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            mlpreg.fit(X, y)
        example = X.iloc[:5, :]

        m = gp.Model()

        x = self.create_input(m, example)

        with self.assertRaises(NoModel):
            add_predictor_constr(m, mlpreg, x)
