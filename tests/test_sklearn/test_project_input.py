"""Tests for ``project_input`` with scikit-learn tree ensembles."""

import unittest

from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from ..tree_ensemble import ProjectInputMixin


class TestProjectInputDecisionTree(ProjectInputMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, y = datasets.load_diabetes(return_X_y=True)
        cls.predictor = DecisionTreeRegressor(max_depth=4, random_state=0).fit(cls.X, y)


class TestProjectInputRandomForest(ProjectInputMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, y = datasets.load_diabetes(return_X_y=True)
        cls.predictor = RandomForestRegressor(
            n_estimators=10, max_depth=4, random_state=0
        ).fit(cls.X, y)


class TestProjectInputGradientBoosting(ProjectInputMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, y = datasets.load_diabetes(return_X_y=True)
        cls.predictor = GradientBoostingRegressor(
            n_estimators=10, max_depth=3, random_state=0
        ).fit(cls.X, y)

    def test_incompatible_leaves_are_nan(self):
        # Known instance: at epsilon=0 the trees of "leaf" take opposite
        # branches at a shared threshold, so no input reaches the leaves.
        import numpy as np

        _, projected = self.solve("leaf", 0.0)
        self.assertTrue(np.isnan(projected).all())


if __name__ == "__main__":
    unittest.main()
