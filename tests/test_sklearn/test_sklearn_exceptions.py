import unittest
import warnings

import gurobipy as gp
import numpy as np
from sklearn import datasets
from sklearn.compose import make_column_transformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    QuantileTransformer,
    StandardScaler,
)
from sklearn.svm import LinearSVR

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import NoModel, ParameterError


class TestUnsuportedSklearn(unittest.TestCase):
    def test_logistic_multiclass(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        logreg = LogisticRegression()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            logreg.fit(X, y)
        example = X[10, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")

        with self.assertRaises(NoModel):
            add_predictor_constr(m, logreg, x)

    def test_logistic_wrongarg(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        y = y == 2
        logreg = LogisticRegression()
        logreg.fit(X, y)
        example = X[10, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")

        with self.assertRaises(ParameterError):
            add_predictor_constr(m, logreg, x, output_type="proba")

    def test_mlpregressor_wrong_act(self):
        data = datasets.load_diabetes()

        X = data.data
        y = data.target

        mlpreg = MLPRegressor([10] * 2, activation="logistic")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            mlpreg.fit(X, y)
        example = X[10, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")

        with self.assertRaises(NoModel):
            add_predictor_constr(m, mlpreg, x)

    def test_pipeline_fail_transformer(self):
        data = datasets.load_diabetes()

        X = data.data
        y = data.target

        mlpreg = make_pipeline(QuantileTransformer(), MLPRegressor([10] * 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            mlpreg.fit(X, y)
        example = X[10, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")

        with self.assertRaises(NoModel):
            add_predictor_constr(m, mlpreg, x)

    def test_polynomial_feature_degree3(self):
        data = datasets.load_diabetes()

        X = data.data
        y = data.target

        mlpreg = make_pipeline(PolynomialFeatures(3), MLPRegressor([10] * 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            mlpreg.fit(X, y)
        example = X[10, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")

        with self.assertRaises(NoModel):
            add_predictor_constr(m, mlpreg, x)

    def test_pipeline_fail_regression(self):
        data = datasets.load_diabetes()

        X = data.data
        y = data.target

        mlpreg = make_pipeline(LinearSVR())
        mlpreg.fit(X, y)
        example = X[10, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")

        with self.assertRaises(NoModel):
            add_predictor_constr(m, mlpreg, x)

    def test_pipeline_good_transformer(self):
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

    def test_pipeline_fail_transformer_idx(self):
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
        example = X[:5, :]

        m = gp.Model()

        x = m.addMVar(example.shape)

        with self.assertRaises(NoModel):
            add_predictor_constr(m, mlpreg, x)
