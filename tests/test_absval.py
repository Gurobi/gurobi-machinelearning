import unittest
import warnings

import gurobipy as gp
import numpy as np
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from gurobi_ml.modeling.neuralnet.activations import ReLU
from gurobi_ml.sklearn import add_mlp_regressor_constr


def build_abs_network():
    """Make a scikit-learn neural network for computing absolute value"""
    # We fakely fit a neural network and then set the correct coefficients by hand
    x = np.arange(-10, 10, 1)
    z = np.abs(x)

    X = np.concatenate([x.ravel().reshape(-1, 1)], axis=1)
    y = z.ravel()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    regression = MLPRegressor(hidden_layer_sizes=[2], max_iter=5, activation="relu", verbose=0)
    regression.fit(X=X, y=y)

    regression.intercepts_ = [np.array([0, 0]), np.array([0])]

    regression.coefs_ = [np.array([[1.0, -1.0]]), np.array([[1.0], [1.0]])]

    checkvals = np.array([-10, -1, 0, 1, 12])
    assert (regression.predict(checkvals.reshape(-1, 1)) == np.abs(checkvals)).all()
    return regression


def absmodel(X, y, nn, infbound, relumodel=None):
    bound = 100
    with gp.Model() as model:
        samples, dim = X.shape
        assert samples == y.shape[0]

        X = np.concatenate([X, np.ones((samples, 1))], axis=1)

        # Decision variables
        beta = model.addMVar(dim + 1, lb=-bound, ub=bound, name="beta")  # Weights
        diff = model.addMVar((samples, 1), lb=-infbound, ub=infbound, name="diff")
        absdiff = model.addMVar((samples, 1), lb=-infbound, ub=infbound, name="absdiff")

        model.addConstr(X @ beta - y == diff[:, 0])
        model.setObjective(absdiff.sum(), gp.GRB.MINIMIZE)

        model.update()

        if nn:
            # create transforms to turn scikit-learn pipeline into Gurobi constraints
            add_mlp_regressor_constr(model, nn, diff, absdiff, activations_models={"relu": relumodel})
        else:
            for i in range(samples):
                model.addConstr(absdiff[i, 0].item() == gp.abs_(diff[i, 0].item()))

        model.Params.OutputFlag = 0
        model.Params.WorkLimit = 100

        model.optimize()
        return model.ObjVal


class TestNNFormulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test"""
        X, y = make_regression(n_samples=30, n_features=3, random_state=0)
        y += np.random.normal(size=(30))
        cls.X = X
        cls.y = y
        cls.nn = build_abs_network()

    def dothetest(self, bound, relu):
        """Test the network for absolute value with relu formulation"""
        val1 = absmodel(self.X, self.y, False, bound, None)
        val2 = absmodel(self.X, self.y, self.nn, bound, relu)
        self.assertAlmostEqual(val1, val2, places=4)

    def test_nobounds(self):
        self.dothetest(gp.GRB.INFINITY, None)

    def test_bounds(self):
        self.dothetest(100, None)

    def test_prop_without_bounds(self):
        """Test with propagation and without bounds"""
        self.dothetest(gp.GRB.INFINITY, ReLU())

    def test_prop_with_bounds(self):
        """Test with propagation and with bounds"""
        self.dothetest(100, ReLU())
