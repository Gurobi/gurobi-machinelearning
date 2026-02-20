"""Tests for PyTorch Softplus activation support."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import gurobipy as gp

from gurobi_ml import add_predictor_constr
from gurobi_ml.exceptions import NoModel

# Check Gurobi version
GUROBI_VERSION = gp.gurobi.version()
HAS_NLFUNC = GUROBI_VERSION >= (12, 0, 0)


class TestPyTorchSoftplus:
    """Test PyTorch Softplus layer support."""

    def test_softplus_basic(self):
        """Test basic Softplus activation with default parameters."""
        # Create a simple model with Softplus
        model = nn.Sequential(
            nn.Linear(2, 3), nn.Softplus(beta=1.0, threshold=20), nn.Linear(3, 1)
        )

        # Set deterministic weights for testing
        with torch.no_grad():
            model[0].weight.data = torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.3, 0.7]])
            model[0].bias.data = torch.tensor([0.1, -0.1, 0.0])
            model[2].weight.data = torch.tensor([[1.0, -0.5, 0.3]])
            model[2].bias.data = torch.tensor([0.2])

        # Test with Gurobi
        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            X_test = np.array([[0.5, 0.3], [1.0, -0.5]])
            x = gpm.addMVar(X_test.shape, lb=X_test - 1e-4, ub=X_test + 1e-4)

            if not HAS_NLFUNC:
                # Expect RuntimeError on older Gurobi versions
                with pytest.raises(
                    RuntimeError,
                    match="SoftReLU requires Gurobi 12.0\\+ with nonlinear function support",
                ):
                    add_predictor_constr(gpm, model, x)
                return

            pred_constr = add_predictor_constr(gpm, model, x)
            gpm.optimize()

            # Compare with PyTorch output
            with torch.no_grad():
                expected = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

            actual = pred_constr.output.X
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    def test_softplus_custom_beta(self):
        """Test Softplus with custom beta parameter."""
        if not HAS_NLFUNC:
            pytest.skip("Requires Gurobi 12.0+ with nonlinear function support")
            
        model = nn.Sequential(
            nn.Linear(2, 2), nn.Softplus(beta=2.0, threshold=20), nn.Linear(2, 1)
        )

        with torch.no_grad():
            model[0].weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            model[0].bias.data = torch.tensor([0.0, 0.0])
            model[2].weight.data = torch.tensor([[1.0, 1.0]])
            model[2].bias.data = torch.tensor([0.0])

        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            X_test = np.array([[1.0, -1.0]])
            x = gpm.addMVar(X_test.shape, lb=X_test - 1e-4, ub=X_test + 1e-4)

            pred_constr = add_predictor_constr(gpm, model, x)
            gpm.optimize()

            with torch.no_grad():
                expected = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

            actual = pred_constr.output.X
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    def test_softplus_invalid_threshold(self):
        """Test that non-default threshold raises NoModel."""
        model = nn.Sequential(
            nn.Linear(2, 2),
            nn.Softplus(beta=1.0, threshold=10),  # Non-default threshold
            nn.Linear(2, 1),
        )

        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            X_test = np.array([[1.0, -1.0]])
            x = gpm.addMVar(X_test.shape, lb=X_test - 1e-4, ub=X_test + 1e-4)

            with pytest.raises(NoModel, match="non-default threshold"):
                add_predictor_constr(gpm, model, x)

    def test_softplus_invalid_beta(self):
        """Test that invalid beta (<=0) raises an error."""
        if not HAS_NLFUNC:
            pytest.skip("Requires Gurobi 12.0+ with nonlinear function support")
            
        # This tests the SoftReLU validation
        from gurobi_ml.modeling.neuralnet.activations import SoftReLU

        with pytest.raises(ValueError, match="beta must be strictly positive"):
            SoftReLU(beta=0.0)

        with pytest.raises(ValueError, match="beta must be strictly positive"):
            SoftReLU(beta=-1.0)
