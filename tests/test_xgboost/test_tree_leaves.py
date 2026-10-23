"""Tests for the ``tree_leaves`` accessor with XGBoost.

With the input fixed to a data point, each tree selects exactly one leaf,
the selected node ids match ``predictor.apply`` (XGBoost's leaf node
numbering equals the booster dump's node ids), and the output matches
``predict``, for the "leaf" baseline and all ensemble formulations.

The fixed point is chosen off every split threshold: XGBoost branches left
iff ``x < t`` (strict) while the formulations encode ``x <= t`` at
epsilon=0, so a row sitting exactly on a threshold may legitimately be
routed to different leaves.
"""

import json
import unittest

import gurobipy as gp
import numpy as np
import xgboost as xgb
from sklearn import datasets

from gurobi_ml import add_predictor_constr


class TestTreeLeavesXGBoost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, y = datasets.load_diabetes(return_X_y=True)
        cls.predictor = xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=0)
        cls.predictor.fit(cls.X, y)
        cls.row = cls._tie_free_row()

    @classmethod
    def _tie_free_row(cls):
        """First row whose float32 coordinates tie no split threshold."""
        raw = json.loads(cls.predictor.get_booster().save_raw("json"))
        thresholds = {}
        for tree in raw["learner"]["gradient_booster"]["model"]["trees"]:
            for feature, threshold, left in zip(
                tree["split_indices"], tree["split_conditions"], tree["left_children"]
            ):
                if left >= 0:
                    thresholds.setdefault(feature, set()).add(
                        float(np.float32(threshold))
                    )
        if not thresholds:
            raise AssertionError("fitted booster has no splits")
        for i, row in enumerate(cls.X.astype(np.float32)):
            if all(float(row[f]) not in ts for f, ts in thresholds.items()):
                return i
        raise AssertionError("no tie-free row in the dataset")

    def selected_matches_apply(self, formulation):
        rows = self.X[self.row : self.row + 1]
        expected = self.predictor.apply(rows)
        with gp.Model() as model:
            model.Params.OutputFlag = 0
            x = model.addMVar(rows.shape, lb=rows, ub=rows)
            pred_constr = add_predictor_constr(
                model, self.predictor, x, formulation=formulation
            )
            tree_leaves = pred_constr.tree_leaves
            self.assertEqual(len(tree_leaves), expected.shape[1])
            model.optimize()
            self.assertEqual(model.Status, gp.GRB.OPTIMAL)
            for t, (variables, nodes) in enumerate(tree_leaves):
                solution = variables.X[0, :]
                self.assertAlmostEqual(solution.sum(), 1.0, places=6)
                self.assertEqual(nodes[int(solution.argmax())], expected[0, t])
            self.assertAlmostEqual(
                pred_constr.output.X[0, 0],
                float(self.predictor.predict(rows)[0]),
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


if __name__ == "__main__":
    unittest.main()
