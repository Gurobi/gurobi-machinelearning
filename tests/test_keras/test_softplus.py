"""Tests for Keras softplus activation support."""

import numpy as np
import pytest
import keras
import gurobipy as gp

from gurobi_ml import add_predictor_constr

# Check Gurobi version
GUROBI_VERSION = gp.gurobi.version()
HAS_NLFUNC = GUROBI_VERSION >= (12, 0, 0)


class TestKerasSoftplus:
    """Test Keras softplus activation support."""

    def test_softplus_activation_in_dense(self):
        """Test Dense layer with softplus activation."""
        if not HAS_NLFUNC:
            pytest.skip("Requires Gurobi 12.0+ with nonlinear function support")
            
        model = keras.Sequential(
            [
                keras.layers.Dense(3, activation="softplus", input_shape=(2,)),
                keras.layers.Dense(1, activation="linear"),
            ]
        )

        # Set deterministic weights
        model.layers[0].set_weights(
            [np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.7]]), np.array([0.1, -0.1, 0.0])]
        )
        model.layers[1].set_weights([np.array([[1.0], [-0.5], [0.3]]), np.array([0.2])])

        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            X_test = np.array([[0.5, 0.3], [1.0, -0.5]])
            x = gpm.addMVar(X_test.shape, lb=X_test - 1e-4, ub=X_test + 1e-4)

            pred_constr = add_predictor_constr(gpm, model, x)
            gpm.optimize()

            expected = model.predict(X_test, verbose=0)
            actual = pred_constr.output.X

            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    def test_softplus_multiple_layers(self):
        """Test model with multiple softplus activations."""
        if not HAS_NLFUNC:
            pytest.skip("Requires Gurobi 12.0+ with nonlinear function support")
            
        model = keras.Sequential(
            [
                keras.layers.Dense(4, activation="softplus", input_shape=(2,)),
                keras.layers.Dense(2, activation="softplus"),
                keras.layers.Dense(1, activation="linear"),
            ]
        )

        # Set deterministic weights
        model.layers[0].set_weights(
            [np.random.RandomState(42).randn(2, 4), np.random.RandomState(42).randn(4)]
        )
        model.layers[1].set_weights(
            [np.random.RandomState(43).randn(4, 2), np.random.RandomState(43).randn(2)]
        )
        model.layers[2].set_weights(
            [np.random.RandomState(44).randn(2, 1), np.random.RandomState(44).randn(1)]
        )

        with (
            gp.Env(params={"OutputFlag": 0, "NonConvex": 2}) as env,
            gp.Model(env=env) as gpm,
        ):
            X_test = np.array([[0.5, 0.3]])
            x = gpm.addMVar(X_test.shape, lb=X_test - 1e-4, ub=X_test + 1e-4)

            pred_constr = add_predictor_constr(gpm, model, x)
            gpm.optimize()

            expected = model.predict(X_test, verbose=0)
            actual = pred_constr.output.X

            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
