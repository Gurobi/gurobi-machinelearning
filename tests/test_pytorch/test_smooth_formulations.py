"""FixedRegressionModel-style tests for smooth activation neural networks.

Mirrors test_pytorch_formulations.py but for smooth (nonlinear) activations:
nn.Sigmoid, nn.Tanh, nn.Softplus.  Each test fixes the input to a Gurobi model,
embeds the network, solves, and verifies the output matches PyTorch's forward pass.
"""

import torch
import torch.nn as nn
import gurobipy as gp
from sklearn import datasets

from ..fixed_formulation import FixedRegressionModel

GUROBI_VERSION = gp.gurobi.version()
HAS_NLFUNC = GUROBI_VERSION >= (12, 0, 0)


def _make_smooth_mlp(activation_cls, n_in, hidden=8, seed=0):
    """Build a tiny smooth MLP with deterministic random weights."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        activation_cls(),
        nn.Linear(hidden, hidden),
        activation_cls(),
        nn.Linear(hidden, 1),
    )


def _case(activation_cls, n_in, seed=0):
    model = _make_smooth_mlp(activation_cls, n_in, seed=seed)
    model.eval()
    return {"predictor": model, "nonconvex": 1}


import unittest


@unittest.skipUnless(HAS_NLFUNC, "Requires Gurobi 12.0+ with nlfunc support")
class TestSmoothActivationFormulations(FixedRegressionModel):
    """Verify smooth-activation network embeddings match PyTorch forward pass."""

    # ------------------------------------------------------------------ sigmoid
    def test_diabetes_sigmoid(self):
        X = datasets.load_diabetes()["data"]
        case = _case(nn.Sigmoid, X.shape[1], seed=1)
        self.do_one_case(case, X, 5, "all")
        self.do_one_case(case, X, 6, "pairs")

    def test_diabetes_sigmoid_wide(self):
        """Wider network (16 neurons) with sigmoid."""
        X = datasets.load_diabetes()["data"]
        torch.manual_seed(2)
        model = nn.Sequential(
            nn.Linear(X.shape[1], 16), nn.Sigmoid(), nn.Linear(16, 1)
        )
        model.eval()
        case = {"predictor": model, "nonconvex": 1}
        self.do_one_case(case, X, 5, "all")
        self.do_one_case(case, X, 6, "pairs")

    # -------------------------------------------------------------------- tanh
    def test_diabetes_tanh(self):
        X = datasets.load_diabetes()["data"]
        case = _case(nn.Tanh, X.shape[1], seed=3)
        self.do_one_case(case, X, 5, "all")
        self.do_one_case(case, X, 6, "pairs")

    def test_diabetes_tanh_wide(self):
        """Wider network (16 neurons) with tanh."""
        X = datasets.load_diabetes()["data"]
        torch.manual_seed(4)
        model = nn.Sequential(
            nn.Linear(X.shape[1], 16), nn.Tanh(), nn.Linear(16, 1)
        )
        model.eval()
        case = {"predictor": model, "nonconvex": 1}
        self.do_one_case(case, X, 5, "all")
        self.do_one_case(case, X, 6, "pairs")

    # ---------------------------------------------------------------- softplus
    def test_diabetes_softplus(self):
        X = datasets.load_diabetes()["data"]
        case = _case(nn.Softplus, X.shape[1], seed=5)
        self.do_one_case(case, X, 5, "all")
        self.do_one_case(case, X, 6, "pairs")

    def test_diabetes_softplus_beta2(self):
        """Softplus with non-default beta=2."""
        X = datasets.load_diabetes()["data"]
        torch.manual_seed(6)
        model = nn.Sequential(
            nn.Linear(X.shape[1], 8),
            nn.Softplus(beta=2, threshold=20),
            nn.Linear(8, 1),
        )
        model.eval()
        case = {"predictor": model, "nonconvex": 1}
        self.do_one_case(case, X, 5, "all")
        self.do_one_case(case, X, 6, "pairs")

    # --------------------------------------------------------- mixed activations
    def test_diabetes_sigmoid_tanh_mixed(self):
        """Network with sigmoid in first layer and tanh in second."""
        X = datasets.load_diabetes()["data"]
        torch.manual_seed(7)
        model = nn.Sequential(
            nn.Linear(X.shape[1], 8),
            nn.Sigmoid(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        model.eval()
        case = {"predictor": model, "nonconvex": 1}
        self.do_one_case(case, X, 5, "all")
        self.do_one_case(case, X, 6, "pairs")
