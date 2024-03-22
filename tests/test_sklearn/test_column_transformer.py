import unittest
import warnings

import gurobipy as gp
import numpy as np
from sklearn import datasets
from sklearn.compose import make_column_transformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import NotRegistered


class TestColumnTransformer(unittest.TestCase):
    """Various test for ColumnTransformerConstr"""

    def test_pipeline_good_transformer_np2dim(self):
        data = datasets.load_diabetes(as_frame=False)

        X = data.data
        y = data.target

        coltran = make_column_transformer(
            (StandardScaler(), [0, 1, 2, 3]),
            ("passthrough", [4, 5, 6, 7, 8, 9]),
        )
        mlpreg = make_pipeline(coltran, MLPRegressor([10] * 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            mlpreg.fit(X, y)
        example = X[:5, :]

        m = gp.Model()

        x = m.addMVar(example.shape)

        add_predictor_constr(m, mlpreg, x)

    def test_pipeline_good_transformer_np1dim(self):
        data = datasets.load_diabetes(as_frame=False)

        X = data.data
        y = data.target

        coltran = make_column_transformer(
            (StandardScaler(), [0, 1, 2, 3]),
            ("passthrough", [4, 5, 6, 7, 8, 9]),
        )
        mlpreg = make_pipeline(coltran, MLPRegressor([10] * 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            mlpreg.fit(X, y)
        example = X[:5, :]

        m = gp.Model()

        x = m.addMVar(example.shape)

        add_predictor_constr(m, mlpreg, x)

    def test_pipeline_fail_transformer_np_1dim(self):
        data = datasets.load_diabetes(as_frame=False)

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
        example = X[5, :]

        m = gp.Model()

        x = m.addMVar(example.shape)

        with self.assertRaises(NotRegistered):
            add_predictor_constr(m, mlpreg, x)
