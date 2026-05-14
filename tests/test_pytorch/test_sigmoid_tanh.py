"""Tests for PyTorch Sigmoid and Tanh activation support."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import gurobipy as gp

from gurobi_ml import add_predictor_constr
from gurobi_ml._grb_version import HAS_NLFUNC, HAS_TANH

# Fixed weights reused across tests
_W0 = torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.3, 0.7]])
_b0 = torch.tensor([0.1, -0.1, 0.0])
_W1 = torch.tensor([[1.0, -0.5, 0.3]])
_b1 = torch.tensor([0.2])
_X = np.array([[0.5, 0.3], [1.0, -0.5]])


def _build_model(act_cls):
    model = nn.Sequential(nn.Linear(2, 3), act_cls(), nn.Linear(3, 1))
    with torch.no_grad():
        model[0].weight.data = _W0.clone()
        model[0].bias.data = _b0.clone()
        model[2].weight.data = _W1.clone()
        model[2].bias.data = _b1.clone()
    return model


def _check_against_torch(act_cls, X=_X):
    model = _build_model(act_cls)
    with (
        gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
        gp.Model(env=env) as gpm,
    ):
        x = gpm.addMVar(X.shape, lb=X - 1e-4, ub=X + 1e-4)
        pc = add_predictor_constr(gpm, model, x)
        gpm.optimize()
        expected = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        np.testing.assert_allclose(pc.output.X, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not HAS_NLFUNC, reason="Requires Gurobi 12.0+ with nlfunc support")
class TestPyTorchSigmoid:
    def test_sigmoid_matches_pytorch(self):
        _check_against_torch(nn.Sigmoid)

    def test_sigmoid_single_sample(self):
        _check_against_torch(nn.Sigmoid, X=np.array([[0.0, 0.0]]))

    def test_sigmoid_negative_inputs(self):
        _check_against_torch(nn.Sigmoid, X=np.array([[-2.0, -1.0], [-0.5, 0.5]]))

    def test_sigmoid_deep_network(self):
        """Two hidden sigmoid layers."""
        model = nn.Sequential(
            nn.Linear(3, 8),
            nn.Sigmoid(),
            nn.Linear(8, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
        )
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.2)
                    nn.init.zeros_(layer.bias)
        X = np.array([[0.5, -0.3, 1.0]])
        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            x = gpm.addMVar(X.shape, lb=X - 1e-4, ub=X + 1e-4)
            pc = add_predictor_constr(gpm, model, x)
            gpm.optimize()
            expected = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
            np.testing.assert_allclose(pc.output.X, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not HAS_NLFUNC, reason="Requires Gurobi 12.0+ with nlfunc support")
@pytest.mark.skipif(not HAS_TANH, reason="Requires Gurobi 13.0+ with tanh support")
class TestPyTorchTanh:
    def test_tanh_matches_pytorch(self):
        _check_against_torch(nn.Tanh)

    def test_tanh_single_sample(self):
        _check_against_torch(nn.Tanh, X=np.array([[0.0, 0.0]]))

    def test_tanh_large_inputs(self):
        """tanh saturates toward ±1; verify embedding handles saturation."""
        _check_against_torch(nn.Tanh, X=np.array([[3.0, -3.0], [5.0, -5.0]]))

    def test_tanh_deep_network(self):
        """Two hidden tanh layers."""
        model = nn.Sequential(
            nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 1)
        )
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.15)
                    nn.init.zeros_(layer.bias)
        X = np.array([[0.5, -0.3, 1.0]])
        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            x = gpm.addMVar(X.shape, lb=X - 1e-4, ub=X + 1e-4)
            pc = add_predictor_constr(gpm, model, x)
            gpm.optimize()
            expected = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
            np.testing.assert_allclose(pc.output.X, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not HAS_TANH, reason="Requires Gurobi 13.0+ with tanh support")
class TestPyTorchMixedActivations:
    def test_sigmoid_then_tanh(self):
        model = nn.Sequential(
            nn.Linear(2, 4), nn.Sigmoid(), nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 1)
        )
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.3)
                    nn.init.zeros_(layer.bias)
        X = np.array([[0.5, -0.5]])
        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            x = gpm.addMVar(X.shape, lb=X - 1e-4, ub=X + 1e-4)
            pc = add_predictor_constr(gpm, model, x)
            gpm.optimize()
            expected = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
            np.testing.assert_allclose(pc.output.X, expected, rtol=1e-3, atol=1e-3)
