"""Tests for ``project_input`` with XGBoost."""

import unittest

import xgboost as xgb
from sklearn import datasets

from ..tree_ensemble import ProjectInputMixin


class TestProjectInputXGBoost(ProjectInputMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, y = datasets.load_diabetes(return_X_y=True)
        cls.predictor = xgb.XGBRegressor(
            n_estimators=10, max_depth=3, random_state=0
        ).fit(cls.X, y)


if __name__ == "__main__":
    unittest.main()
