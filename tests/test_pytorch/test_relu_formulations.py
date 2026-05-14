"""Tests for relu_formulation parameter and smooth/soft ReLU variants."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import gurobipy as gp

from gurobi_ml import add_predictor_constr
from gurobi_ml._grb_version import HAS_NLFUNC
from gurobi_ml.modeling.neuralnet.activations import SoftPlus


@pytest.mark.skipif(not HAS_NLFUNC, reason="Requires Gurobi 12.0+ with nlfunc support")
class TestReLUFormulations:
    """Test different ReLU formulation options."""

    def test_soft_relu_formulation(self):
        """Test relu_formulation='soft' parameter."""
        model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))

        with torch.no_grad():
            model[0].weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
            model[0].bias.data = torch.tensor([0.0, 0.0, 0.0])
            model[2].weight.data = torch.tensor([[1.0, 1.0, 1.0]])
            model[2].bias.data = torch.tensor([0.0])

        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            X_test = np.array([[1.0, -1.0]])
            x = gpm.addMVar(X_test.shape, lb=X_test - 1e-4, ub=X_test + 1e-4)

            pred_constr = add_predictor_constr(
                gpm, model, x, relu_formulation="soft", soft_relu_beta=10.0
            )
            gpm.optimize()

            with torch.no_grad():
                expected = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

            actual = pred_constr.output.X
            # SoftReLU approximates ReLU; with high beta it should be close
            np.testing.assert_allclose(actual, expected, rtol=0.1, atol=0.1)

    def test_invalid_relu_formulation(self):
        """Test that invalid relu_formulation raises ValueError."""
        model = nn.Sequential(nn.Linear(2, 1), nn.ReLU())

        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, 2))

            with pytest.raises(ValueError, match="relu_formulation must be"):
                add_predictor_constr(gpm, model, x, relu_formulation="invalid")

    def test_soft_relu_class_directly(self):
        """Test SoftReLU class with valid beta."""
        soft_relu = SoftPlus(beta=1.0)
        assert soft_relu.beta == 1.0

        soft_relu2 = SoftPlus(beta=5.0)
        assert soft_relu2.beta == 5.0

    def test_soft_relu_invalid_beta(self):
        """Test SoftReLU rejects invalid beta values."""
        with pytest.raises(ValueError, match="beta must be strictly positive"):
            SoftPlus(beta=0.0)

        with pytest.raises(ValueError, match="beta must be strictly positive"):
            SoftPlus(beta=-1.5)
