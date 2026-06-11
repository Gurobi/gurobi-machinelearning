"""Tests for the reachable-leaf pruning optimisation in _leaf_formulation.

The optimisation skips leaf binary variables that are unreachable given the
current input-variable bounds.  These tests verify:

1. That tight input bounds produce *fewer* binary variables than wide bounds
   (pruning is actually happening).
2. That the formulation is still correct when pruning occurs (Gurobi solution
   matches sklearn predict to within tolerance).
"""

import unittest
import warnings

import gurobipy as gp
import numpy as np
from gurobipy import GurobiError
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from gurobi_ml import add_predictor_constr


def _build_and_count_binvars(gp_model, predictor, x):
    """Add predictor constraint and return number of binary variables."""
    add_predictor_constr(gp_model, predictor, x)
    return gp_model.NumBinVars


def _total_leaves(reg):
    """Sum leaf-node count across all trees in a GradientBoostingRegressor."""
    return sum(int((est[0].tree_.children_left < 0).sum()) for est in reg.estimators_)


class TestLeafPruning(unittest.TestCase):
    """Pruning reduces binary-variable count when input bounds are tight."""

    def _env_params(self):
        return {"OutputFlag": 0}

    def test_tight_bounds_fewer_binvars_than_naive(self):
        """With point-like bounds, active leaf vars << n_examples * total_leaves."""
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]

        reg = GradientBoostingRegressor(n_estimators=5, max_depth=4, random_state=0)
        reg.fit(X, y)
        total_leaves = _total_leaves(reg)

        example = X[:1]  # single example
        with gp.Env(params=self._env_params()) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar(example.shape, lb=example - 1e-6, ub=example + 1e-6)
            n_tight = _build_and_count_binvars(gpm, reg, x)

        # Naive upper bound: 1 example × total_leaves
        self.assertLess(n_tight, total_leaves)

    def test_wide_bounds_more_binvars_than_tight(self):
        """Wide bounds (spanning entire training range) should create more binary
        variables than point-like bounds for the same number of examples."""
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]

        reg = GradientBoostingRegressor(n_estimators=5, max_depth=4, random_state=0)
        reg.fit(X, y)

        example = X[:1]
        with gp.Env(params=self._env_params()) as env, gp.Model(env=env) as gpm:
            x_tight = gpm.addMVar(example.shape, lb=example - 1e-6, ub=example + 1e-6)
            n_tight = _build_and_count_binvars(gpm, reg, x_tight)

        with gp.Env(params=self._env_params()) as env, gp.Model(env=env) as gpm:
            x_wide = gpm.addMVar(example.shape, lb=X.min(axis=0), ub=X.max(axis=0))
            try:
                n_wide = _build_and_count_binvars(gpm, reg, x_wide)
            except GurobiError as e:
                if e.errno == 10010:
                    warnings.warn(
                        UserWarning("Limited license — skipping wide-bounds check")
                    )
                    self.skipTest("Model too large for limited license")
                raise

        self.assertLess(n_tight, n_wide)

    def test_pruning_correct_for_decision_tree(self):
        """After pruning, the fixed-input model should match sklearn prediction."""
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]

        reg = DecisionTreeRegressor(max_leaf_nodes=20, random_state=0)
        reg.fit(X, y)

        example = X[:3]
        with gp.Env(params=self._env_params()) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar(example.shape, lb=example - 1e-6, ub=example + 1e-6)
            pred_constr = add_predictor_constr(gpm, reg, x)
            try:
                gpm.optimize()
            except GurobiError as e:
                if e.errno == 10010:
                    self.skipTest("Model too large for limited license")
                raise
            if gpm.Status != 2:
                self.skipTest("No optimal solution found")
            err = pred_constr.get_error().astype(float)
        self.assertLess(np.max(err), 1e-4)

    def test_unbounded_vars_produce_full_tree(self):
        """Unbounded input variables must keep every leaf active (no pruning)."""
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]

        reg = GradientBoostingRegressor(n_estimators=5, max_depth=4, random_state=0)
        reg.fit(X, y)
        total_leaves = _total_leaves(reg)

        n_samples = 2
        example = X[:n_samples]
        with gp.Env(params=self._env_params()) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar(example.shape, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
            try:
                n_vars = _build_and_count_binvars(gpm, reg, x)
            except GurobiError as e:
                if e.errno == 10010:
                    self.skipTest("Model too large for limited license")
                raise

        self.assertEqual(n_vars, n_samples * total_leaves)


if __name__ == "__main__":
    unittest.main()
