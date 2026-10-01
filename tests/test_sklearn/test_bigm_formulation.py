# Copyright © 2023-2026 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for the big-M formulations (formulation="bigm")."""

import unittest

import gurobipy as gp
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from gurobi_ml import add_predictor_constr


class TestBigMFormulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = load_diabetes(return_X_y=True)
        cls.X = X
        cls.gbt = GradientBoostingRegressor(
            n_estimators=5, max_depth=3, random_state=0
        ).fit(X, y)
        cls.mlp = MLPRegressor(
            hidden_layer_sizes=[10, 10], max_iter=500, random_state=0
        ).fit(X, y)

    def setUp(self):
        self.env = gp.Env(params={"OutputFlag": 0})
        self.model = gp.Model(env=self.env)

    def tearDown(self):
        self.model.dispose()
        self.env.dispose()

    def test_no_general_constraints(self):
        for predictor in (self.gbt, self.mlp):
            with self.subTest(predictor=type(predictor).__name__):
                m = gp.Model(env=self.env)
                x = m.addMVar((3, self.X.shape[1]), lb=-1.0, ub=1.0)
                add_predictor_constr(m, predictor, x, formulation="bigm")
                m.update()
                self.assertEqual(m.NumGenConstrs, 0)
                self.assertGreater(m.NumBinVars, 0)

    def test_infinite_bounds_require_bigm(self):
        for predictor in (self.gbt, self.mlp):
            with self.subTest(predictor=type(predictor).__name__):
                m = gp.Model(env=self.env)
                x = m.addMVar((1, self.X.shape[1]), lb=-gp.GRB.INFINITY)
                with self.assertRaises(ValueError):
                    add_predictor_constr(m, predictor, x, formulation="bigm")

    def test_bigm_value_for_infinite_bounds(self):
        # Input unbounded but fixed by an equality constraint: the bigm value
        # is used in the formulation and the prediction should be exact.
        example = self.X[:2, :]
        for predictor in (self.gbt, self.mlp):
            with self.subTest(predictor=type(predictor).__name__):
                m = gp.Model(env=self.env)
                x = m.addMVar(example.shape, lb=-gp.GRB.INFINITY)
                m.addConstr(x == example)
                pred_constr = add_predictor_constr(
                    m, predictor, x, formulation="bigm", bigm=1e3
                )
                m.optimize()
                self.assertEqual(m.NumGenConstrs, 0)
                self.assertLessEqual(np.max(pred_constr.get_error()), 1e-5)

    def test_user_output_bounds_unchanged(self):
        for predictor in (self.gbt, self.mlp):
            with self.subTest(predictor=type(predictor).__name__):
                m = gp.Model(env=self.env)
                x = m.addMVar((2, self.X.shape[1]), lb=-1.0, ub=1.0)
                y = m.addMVar((2, 1), lb=-gp.GRB.INFINITY)
                m.update()
                add_predictor_constr(m, predictor, x, y, formulation="bigm")
                m.update()
                self.assertTrue(np.all(y.LB <= -gp.GRB.INFINITY))
                self.assertTrue(np.all(y.UB >= gp.GRB.INFINITY))

    def test_unknown_formulation(self):
        x = self.model.addMVar((1, self.X.shape[1]), lb=-1.0, ub=1.0)
        with self.assertRaises(ValueError):
            add_predictor_constr(self.model, self.mlp, x, formulation="unknown")

    def test_uniform_bigm(self):
        # With a given bigm all hidden neurons get a binary variable and the
        # prediction is still correct when the input is fixed
        example = self.X[:3, :]
        n_neurons = sum(self.mlp.hidden_layer_sizes) * example.shape[0]
        for bigm, n_binaries in ((None, None), (1e3, n_neurons)):
            with self.subTest(bigm=bigm):
                m = gp.Model(env=self.env)
                x = m.addMVar(example.shape, lb=example - 1e-4, ub=example + 1e-4)
                pred_constr = add_predictor_constr(
                    m, self.mlp, x, formulation="bigm", bigm=bigm
                )
                m.optimize()
                if n_binaries is None:
                    self.assertLess(m.NumBinVars, n_neurons)
                else:
                    self.assertEqual(m.NumBinVars, n_binaries)
                self.assertLessEqual(np.max(pred_constr.get_error()), 1e-4)
