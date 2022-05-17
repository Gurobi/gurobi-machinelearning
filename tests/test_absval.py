import unittest

import gurobipy as gp
import numpy as np
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor

from ml2gurobi.activations2gurobi import ReLUGC
from ml2gurobi.extra.morerelu import ReLUmin
from ml2gurobi.sklearn2gurobi import MLPRegressor2Gurobi


def build_abs_network():
    ''' Make a scikit-learn neural network for computing absolute value'''
    # We fakely fit a neural network and then set the correct coefficients by hand
    x = np.arange(-10, 10, 1)
    z = np.abs(x)

    X = np.concatenate([x.ravel().reshape(-1, 1)], axis=1)
    y = z.ravel()

    regression = MLPRegressor(hidden_layer_sizes=[2], max_iter=5, activation='relu', verbose=0)
    regression.fit(X=X, y=y)

    regression.intercepts_ = [np.array([0, 0]), np.array([0])]

    regression.coefs_ = [np.array([[1.0, -1.0]]),
                         np.array([[1.0], [1.0]])]

    checkvals = np.array([-10, -1, 0, 1, 12])
    assert (regression.predict(checkvals.reshape(-1, 1)) == np.abs(checkvals)).all()
    return regression


def model(X, y, nn, infbound, relumodel=None):
    bound = 100
    with gp.Model() as regressor:
        samples, dim = X.shape
        assert samples == y.shape[0]

        X = np.concatenate([X, np.ones((samples, 1))], axis=1)

        # Decision variables
        beta = regressor.addMVar(dim + 1, lb=-bound, ub=bound, name="beta")  # Weights
        diff = regressor.addMVar((samples, 1), lb=-infbound, ub=infbound, name='diff')
        absdiff = regressor.addMVar((samples, 1), lb=-infbound, ub=infbound, name='absdiff')

        regressor.addConstr(X @ beta - y == diff[:, 0])
        regressor.setObjective(absdiff.sum(), gp.GRB.MINIMIZE)

        if nn:
            # create transforms to turn scikit-learn pipeline into Gurobi constraints
            nn2gurobi = MLPRegressor2Gurobi(nn, regressor)
            if relumodel is not None:
                nn2gurobi.actdict['relu'] = relumodel

            # Add constraint to predict value of y using kwnown and to compute features
            nn2gurobi.predict(X=diff, y=absdiff)
        else:
            for i in range(samples):
                regressor.addConstr(absdiff[i, 0] == gp.abs_(diff[i, 0]))

        regressor.Params.OutputFlag = 0
        regressor.Params.WorkLimit = 100

        regressor.optimize()
        return regressor.ObjVal


class TestNNFormulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        X, y = make_regression(n_samples=30, n_features=3, random_state=0)
        y += np.random.normal(size=(30))
        cls.X = X
        cls.y = y
        cls.nn = build_abs_network()

    def dothetest(self, b, r):
        val1 = model(self.X, self.y, False, b, None)
        val2 = model(self.X, self.y, self.nn, b, r)
        self.assertAlmostEqual(val1, val2)

    def test_nobounds(self):
        self.dothetest(gp.GRB.INFINITY, None)

    def test_bounds(self):
        self.dothetest(100, None)

    def test_prop_without_bounds(self):
        self.dothetest(gp.GRB.INFINITY, ReLUGC(setbounds=True))

    def test_prop_with_bounds(self):
        self.dothetest(100, ReLUGC(setbounds=True))

    def test_noprop_without_bounds(self):
        self.dothetest(gp.GRB.INFINITY, ReLUGC(setbounds=False))

    def test_noprop_with_bounds(self):
        self.dothetest(100, ReLUGC(setbounds=False))

    def test_min_prop_without_bounds(self):
        self.dothetest(gp.GRB.INFINITY, ReLUmin(setbounds=True))

    def test_min_prop_with_bounds(self):
        self.dothetest(100, ReLUmin(setbounds=True))

    def test_min_noprop_without_bounds(self):
        self.dothetest(gp.GRB.INFINITY, ReLUmin(setbounds=False))

    def test_min_noprop_with_bounds(self):
        self.dothetest(100, ReLUmin(setbounds=False))
