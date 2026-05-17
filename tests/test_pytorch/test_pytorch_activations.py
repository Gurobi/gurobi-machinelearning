"""Tests for PyTorch activation function support in Gurobi ML.

Covers Sigmoid, Tanh, Softplus (parametrised), and ReLU formulation options.
All tests use synthetic fixed-weight networks — no external data required.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import gurobipy as gp

from gurobi_ml import add_predictor_constr
from gurobi_ml._grb_version import HAS_NLFUNC, HAS_TANH
from gurobi_ml.exceptions import ModelConfigurationError
from gurobi_ml.modeling.neuralnet.activations import SoftPlus

# Fixed weights for reproducible shallow 2→3→1 networks.
_W0 = torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.3, 0.7]])
_b0 = torch.tensor([0.1, -0.1, 0.0])
_W1 = torch.tensor([[1.0, -0.5, 0.3]])
_b1 = torch.tensor([0.2])


def _build_shallow(act_cls):
    """Build a 2→3→1 MLP with fixed weights and the given activation."""
    model = nn.Sequential(nn.Linear(2, 3), act_cls(), nn.Linear(3, 1))
    with torch.no_grad():
        model[0].weight.data = _W0.clone()
        model[0].bias.data = _b0.clone()
        model[2].weight.data = _W1.clone()
        model[2].bias.data = _b1.clone()
    return model


def _check_against_torch(model, X, **kwargs):
    """Pin input X in a Gurobi model, solve, and assert output matches PyTorch."""
    with (
        gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
        gp.Model(env=env) as gpm,
    ):
        x = gpm.addMVar(X.shape, lb=X - 1e-4, ub=X + 1e-4)
        pc = add_predictor_constr(gpm, model, x, **kwargs)
        gpm.optimize()
        expected = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        np.testing.assert_allclose(pc.output.X, expected, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# Shared skip markers                                                          #
# --------------------------------------------------------------------------- #

_nlfunc_skip = pytest.mark.skipif(
    not HAS_NLFUNC, reason="Requires Gurobi 12.0+ with nlfunc support"
)
_tanh_skip = pytest.mark.skipif(
    not HAS_TANH, reason="Requires Gurobi 13.0+ with tanh support"
)

# Parametrised activation list: relu, sigmoid, tanh, softplus.
_ACTIVATIONS = [
    pytest.param(nn.ReLU, marks=[], id="relu"),
    pytest.param(nn.Sigmoid, marks=_nlfunc_skip, id="sigmoid"),
    pytest.param(nn.Tanh, marks=[_nlfunc_skip, _tanh_skip], id="tanh"),
    pytest.param(nn.Softplus, marks=_nlfunc_skip, id="softplus"),
]


# --------------------------------------------------------------------------- #
# Parametrised activation tests (identical logic, one class per activation)   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("act_cls", _ACTIVATIONS)
class TestActivationMatchesPyTorch:
    """Gurobi embedding output must match PyTorch for every supported activation."""

    def test_standard_input(self, act_cls):
        _check_against_torch(
            _build_shallow(act_cls), np.array([[0.5, 0.3], [1.0, -0.5]])
        )

    def test_single_sample(self, act_cls):
        _check_against_torch(_build_shallow(act_cls), np.array([[0.0, 0.0]]))

    def test_negative_inputs(self, act_cls):
        _check_against_torch(
            _build_shallow(act_cls), np.array([[-2.0, -1.0], [-0.5, 0.5]])
        )

    def test_deep_network(self, act_cls):
        """Two hidden layers with the same activation."""
        model = nn.Sequential(
            nn.Linear(3, 8), act_cls(), nn.Linear(8, 4), act_cls(), nn.Linear(4, 1)
        )
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.2)
                    nn.init.zeros_(layer.bias)
        _check_against_torch(model, np.array([[0.5, -0.3, 1.0]]))

    def test_wide_network(self, act_cls):
        """Single wide hidden layer (16 neurons) with the same activation."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 16), act_cls(), nn.Linear(16, 1))
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.3)
                    nn.init.zeros_(layer.bias)
        _check_against_torch(model, np.array([[0.5, -0.3, 1.0, 0.2]]))


# --------------------------------------------------------------------------- #
# Mixed-activation network                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_NLFUNC, reason="Requires Gurobi 12.0+ with nlfunc support")
@pytest.mark.skipif(not HAS_TANH, reason="Requires Gurobi 13.0+ with tanh support")
class TestMixedActivations:
    def test_sigmoid_then_tanh(self):
        model = nn.Sequential(
            nn.Linear(2, 4), nn.Sigmoid(), nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 1)
        )
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.3)
                    nn.init.zeros_(layer.bias)
        _check_against_torch(model, np.array([[0.5, -0.5]]))


# --------------------------------------------------------------------------- #
# ReLU formulation options                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_NLFUNC, reason="Requires Gurobi 12.0+ with nlfunc support")
class TestReLUFormulations:
    def test_soft_relu_formulation(self):
        """relu_formulation='soft' should approximate ReLU closely at high beta."""
        model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
        with torch.no_grad():
            model[0].weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
            model[0].bias.data = torch.zeros(3)
            model[2].weight.data = torch.ones(1, 3)
            model[2].bias.data = torch.zeros(1)

        X = np.array([[1.0, -1.0]])
        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            x = gpm.addMVar(X.shape, lb=X - 1e-4, ub=X + 1e-4)
            pc = add_predictor_constr(
                gpm, model, x, relu_formulation="soft", soft_relu_beta=10.0
            )
            gpm.optimize()
            expected = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
            # SoftReLU approximates ReLU; high beta keeps the error small.
            np.testing.assert_allclose(pc.output.X, expected, rtol=0.1, atol=0.1)

    def test_invalid_relu_formulation(self):
        model = nn.Sequential(nn.Linear(2, 1), nn.ReLU())
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, 2))
            with pytest.raises(ValueError, match="relu_formulation must be"):
                add_predictor_constr(gpm, model, x, relu_formulation="invalid")


# --------------------------------------------------------------------------- #
# Softplus-specific configuration and error handling                           #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_NLFUNC, reason="Requires Gurobi 12.0+ with nlfunc support")
class TestSoftPlusClass:
    """Unit tests for the SoftPlus wrapper class (no Gurobi needed)."""

    def test_valid_beta(self):
        assert SoftPlus(beta=1.0).beta == 1.0
        assert SoftPlus(beta=5.0).beta == 5.0

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="beta must be strictly positive"):
            SoftPlus(beta=0.0)
        with pytest.raises(ValueError, match="beta must be strictly positive"):
            SoftPlus(beta=-1.5)


@pytest.mark.skipif(not HAS_NLFUNC, reason="Requires Gurobi 12.0+ with nlfunc support")
class TestSoftplusConfiguration:
    def test_custom_beta(self):
        """Non-default beta should still produce correct Gurobi output."""
        model = nn.Sequential(
            nn.Linear(2, 2), nn.Softplus(beta=2.0, threshold=20), nn.Linear(2, 1)
        )
        with torch.no_grad():
            model[0].weight.data = torch.eye(2)
            model[0].bias.data = torch.zeros(2)
            model[2].weight.data = torch.ones(1, 2)
            model[2].bias.data = torch.zeros(1)
        _check_against_torch(model, np.array([[1.0, -1.0]]))

    def test_invalid_threshold(self):
        """Non-default threshold value must raise ModelConfigurationError."""
        model = nn.Sequential(
            nn.Linear(2, 2), nn.Softplus(beta=1.0, threshold=10), nn.Linear(2, 1)
        )
        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            x = gpm.addMVar((1, 2))
            with pytest.raises(ModelConfigurationError, match="non-default threshold"):
                add_predictor_constr(gpm, model, x)
