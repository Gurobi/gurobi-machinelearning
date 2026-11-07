"""Tests for the ``tree_leaves`` accessor with LightGBM.

With the input fixed to a data point, each tree selects exactly one leaf,
the selected node ids match ``predict(pred_leaf=True)`` (the flat tree
representation numbers a leaf as ``leaf_index + num_splits``, and a binary
tree has ``num_leaves - 1`` splits), and the output matches ``predict``,
for the "leaf" baseline and all ensemble formulations.
"""

import unittest

import gurobipy as gp
import lightgbm as lgb
from sklearn import datasets

from gurobi_ml import add_predictor_constr


class TestTreeLeavesLightGBM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, y = datasets.load_diabetes(return_X_y=True)
        cls.predictor = lgb.LGBMRegressor(
            n_estimators=5, max_depth=3, random_state=0, verbose=-1
        )
        cls.predictor.fit(cls.X, y)

    def selected_matches_pred_leaf(self, formulation):
        rows = self.X[7:8]
        leaf_index = self.predictor.predict(rows, pred_leaf=True)
        trees = self.predictor.booster_.dump_model()["tree_info"]
        expected = [
            leaf_index[0, t] + tree["num_leaves"] - 1 for t, tree in enumerate(trees)
        ]
        with gp.Model() as model:
            model.Params.OutputFlag = 0
            x = model.addMVar(rows.shape, lb=rows, ub=rows)
            pred_constr = add_predictor_constr(
                model, self.predictor, x, formulation=formulation
            )
            tree_leaves = pred_constr.tree_leaves
            self.assertEqual(len(tree_leaves), len(trees))
            model.optimize()
            self.assertEqual(model.Status, gp.GRB.OPTIMAL)
            for t, (variables, nodes) in enumerate(tree_leaves):
                solution = variables.X[0, :]
                self.assertAlmostEqual(solution.sum(), 1.0, places=6)
                self.assertEqual(nodes[int(solution.argmax())], expected[t])
            self.assertAlmostEqual(
                pred_constr.output.X[0, 0],
                float(self.predictor.predict(rows)[0]),
                places=4,
            )

    def test_selected_matches_pred_leaf_leaf(self):
        self.selected_matches_pred_leaf("leaf")

    def test_selected_matches_pred_leaf_misic(self):
        self.selected_matches_pred_leaf("misic")

    def test_selected_matches_pred_leaf_ocean(self):
        self.selected_matches_pred_leaf("ocean")

    def test_selected_matches_pred_leaf_biggs_perakis(self):
        self.selected_matches_pred_leaf("biggs_perakis")


if __name__ == "__main__":
    unittest.main()
