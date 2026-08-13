import unittest

import gurobipy as gp
import lightgbm as lgb
import numpy as np
from gurobipy import GRB

from gurobi_ml.lightgbm import add_lgbm_booster_constr


class TestIssue496(unittest.TestCase):
    """Test case for issue #496: numerical issues with sparse features in LightGBM."""

    def setUp(self):
        # Training data: feature 0 has many EXACT zeros + some non-zero active values
        np.random.seed(42)
        N = 1000
        X = np.random.randn(N, 5).astype(np.float32)
        X[:, 0] = 0.0
        active = np.random.rand(N) < 0.05
        X[active, 0] = np.random.choice([-5, 5], size=active.sum()).astype(np.float32)
        y = (X[:, 0] != 0).astype(np.int32)

        self.model = lgb.train(
            {
                "objective": "binary",
                "num_leaves": 4,
                "max_depth": 2,
                "learning_rate": 0.3,
                "verbose": -1,
                "seed": 42,
                "deterministic": True,
                "min_data_in_leaf": 10,
            },
            lgb.Dataset(X, y),
            num_boost_round=1,
        )
        self.X = X

    def test_numerical_issue_leaf_formulation(self):
        self._check_numerical_issue_warning("leaf")

    def test_numerical_issue_misic(self):
        self._check_numerical_issue_warning("misic")

    def test_numerical_issue_ocean(self):
        self._check_numerical_issue_warning("ocean")

    def test_numerical_issue_biggs_perakis(self):
        self._check_numerical_issue_warning("biggs_perakis")

    def _check_numerical_issue_warning(self, formulation):
        """Verify that a warning is raised when thresholds fall below FeasibilityTol."""
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            x_vars = m.addMVar(shape=5, lb=-10.0, ub=10.0, name="x")
            y_var = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="score")
            with self.assertWarns(UserWarning):
                add_lgbm_booster_constr(
                    m, self.model, x_vars, y_var, formulation=formulation
                )

    def test_leaf_formulation_fix(self):
        self._check_safety_floor_fix("leaf")

    def test_misic_formulation_fix(self):
        self._check_safety_floor_fix("misic")

    def test_ocean_formulation_fix(self):
        self._check_safety_floor_fix("ocean")

    def test_biggs_perakis_formulation_fix(self):
        self._check_safety_floor_fix("biggs_perakis")

    def _check_safety_floor_fix(self, formulation):
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            x_vars = m.addMVar(shape=5, lb=-10.0, ub=10.0, name="x")
            y_var = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="score")

            # Use the formulation with safety_floor to avoid the issue
            add_lgbm_booster_constr(
                m,
                self.model,
                x_vars,
                y_var,
                formulation=formulation,
                safety_floor=1e-5,
            )
            m.update()

            # Test points: 0.0 should be "zero", 1e-3 should be "non-zero" (well above safety_floor=1e-5)
            for x0 in [0.0, 1e-3, 1.0, 5.0]:
                x_test = np.array([x0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                lgbm_score = float(
                    self.model.predict(x_test.reshape(1, -1), raw_score=True)[0]
                )

                for j in range(5):
                    x_vars[j].LB = float(x_test[j])
                    x_vars[j].UB = float(x_test[j])

                m.optimize()
                self.assertEqual(m.Status, GRB.OPTIMAL)
                mip_score = float(y_var.X)

                self.assertAlmostEqual(
                    lgbm_score,
                    mip_score,
                    places=5,
                    msg=f"Mismatch at x0={x0}: lgbm={lgbm_score}, mip={mip_score}",
                )
