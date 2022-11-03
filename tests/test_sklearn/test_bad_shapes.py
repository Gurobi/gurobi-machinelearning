import unittest

import gurobipy as gp
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import ParameterError


class TestBadShapes(unittest.TestCase):
    def test_sould_work(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        y = y == 2
        logreg = LogisticRegression()
        logreg.fit(X, y)
        example = X[10:15, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")
        y = m.addMVar(example.shape[0], name="y")

        add_predictor_constr(m, logreg, x, y)

    def test_3d_shape(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        y = y == 2
        logreg = LogisticRegression()
        logreg.fit(X, y)
        example = X[10:15, :]

        m = gp.Model()

        (a, b) = example.shape
        x = m.addMVar((a, b, 1), name="x")
        y = m.addMVar(example.shape[0], name="y")

        with self.assertRaises(ParameterError):
            add_predictor_constr(m, logreg, x, y)

    def test_mismatch_first_dim(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        y = y == 2
        logreg = LogisticRegression()
        logreg.fit(X, y)
        example = X[10:15, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")
        y = m.addMVar(2, name="y")

        with self.assertRaises(ParameterError):
            add_predictor_constr(m, logreg, x, y)

    def test_input_not_vars(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        y = y == 2
        logreg = LogisticRegression()
        logreg.fit(X, y)
        example = X[10:15, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")
        y = m.addMVar(example.shape[0], name="y")

        with self.assertRaises(ParameterError):
            add_predictor_constr(m, logreg, x == 1, y)

    def test_output_not_vars(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        y = y == 2
        logreg = LogisticRegression()
        logreg.fit(X, y)
        example = X[10:15, :]

        m = gp.Model()

        x = m.addMVar(example.shape, name="x")
        y = m.addMVar(example.shape[0], name="y")

        with self.assertRaises(ParameterError):
            add_predictor_constr(m, logreg, x, y == 1)

    def test_empty_input(self):
        data = datasets.load_iris()

        X = data.data
        y = data.target

        y = y == 2
        logreg = LogisticRegression()
        logreg.fit(X, y)
        example = X[10:15, :]

        m = gp.Model()

        with self.assertRaises(ParameterError):
            add_predictor_constr(m, logreg, None)
