"""Tests for the public per-tree leaf variable accessor ``tree_leaves``.

For every formulation ("leaf" baseline and the four ensemble formulations)
and every sklearn tree predictor type: the accessor has one entry per tree,
and with the input fixed to a data point each tree selects exactly one leaf,
the selected node ids match ``predictor.apply`` and the output matches
``predict``. Multiple input rows are supported (one row of leaf variables
per input row), and the deprecated "paths" formulation raises.
"""

import unittest

import gurobipy as gp
import numpy as np
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from gurobi_ml import add_predictor_constr


def _apply_2d(predictor, X):
    """Leaf node id per (example, tree), shape (n, n_trees)."""
    leaf_of = predictor.apply(X)
    if leaf_of.ndim == 1:
        return leaf_of.reshape(-1, 1)
    if leaf_of.ndim == 3:
        return leaf_of[:, :, 0]
    return leaf_of


class TestTreeLeaves(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, y = datasets.load_diabetes(return_X_y=True)
        cls.predictors = {
            "gbt": GradientBoostingRegressor(
                n_estimators=8, max_depth=3, random_state=0
            ),
            "rf": RandomForestRegressor(n_estimators=8, max_depth=3, random_state=0),
            "dt": DecisionTreeRegressor(max_depth=4, random_state=0),
        }
        for predictor in cls.predictors.values():
            predictor.fit(cls.X, y)

    def selected_matches_apply(self, formulation):
        """Fixed input: selection equals apply, output equals predict."""
        rows = self.X[7:8]
        for name, predictor in self.predictors.items():
            with self.subTest(predictor=name):
                with gp.Model() as model:
                    model.Params.OutputFlag = 0
                    x = model.addMVar(rows.shape, lb=rows, ub=rows)
                    pred_constr = add_predictor_constr(
                        model, predictor, x, formulation=formulation
                    )
                    tree_leaves = pred_constr.tree_leaves
                    expected = _apply_2d(predictor, rows)
                    self.assertEqual(len(tree_leaves), expected.shape[1])
                    model.optimize()
                    self.assertEqual(model.Status, gp.GRB.OPTIMAL)
                    for t, (variables, nodes) in enumerate(tree_leaves):
                        solution = variables.X[0, :]
                        self.assertAlmostEqual(solution.sum(), 1.0, places=6)
                        self.assertEqual(nodes[int(solution.argmax())], expected[0, t])
                    self.assertAlmostEqual(
                        pred_constr.output.X[0, 0],
                        predictor.predict(rows)[0],
                        places=4,
                    )

    def test_selected_matches_apply_leaf(self):
        self.selected_matches_apply("leaf")

    def test_selected_matches_apply_misic(self):
        self.selected_matches_apply("misic")

    def test_selected_matches_apply_ocean(self):
        self.selected_matches_apply("ocean")

    def test_selected_matches_apply_biggs_perakis(self):
        self.selected_matches_apply("biggs_perakis")

    def multiple_input_rows(self, formulation):
        """One row of leaf variables per input row."""
        rows = self.X[:5]
        predictor = self.predictors["gbt"]
        expected = _apply_2d(predictor, rows)
        with gp.Model() as model:
            model.Params.OutputFlag = 0
            x = model.addMVar(rows.shape, lb=rows, ub=rows)
            pred_constr = add_predictor_constr(
                model, predictor, x, formulation=formulation
            )
            model.optimize()
            self.assertEqual(model.Status, gp.GRB.OPTIMAL)
            for t, (variables, nodes) in enumerate(pred_constr.tree_leaves):
                chosen = nodes[variables.X.argmax(axis=1)]
                np.testing.assert_array_equal(chosen, expected[:, t])

    def test_multiple_input_rows_leaf(self):
        self.multiple_input_rows("leaf")

    def test_multiple_input_rows_misic(self):
        self.multiple_input_rows("misic")

    def test_multiple_input_rows_ocean(self):
        self.multiple_input_rows("ocean")

    def test_multiple_input_rows_biggs_perakis(self):
        self.multiple_input_rows("biggs_perakis")

    def test_paths_formulation_raises(self):
        with gp.Model() as model:
            x = model.addMVar(
                (1, self.X.shape[1]), lb=self.X.min(axis=0), ub=self.X.max(axis=0)
            )
            pred_constr = add_predictor_constr(
                model, self.predictors["dt"], x, formulation="paths"
            )
            with self.assertRaises(AttributeError):
                pred_constr.tree_leaves


if __name__ == "__main__":
    unittest.main()
