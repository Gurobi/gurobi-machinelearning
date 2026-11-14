"""Tests for ``project_input`` with LightGBM."""

import unittest

import lightgbm as lgb
from sklearn import datasets

from ..tree_ensemble import ProjectInputMixin


class TestProjectInputLightGBM(ProjectInputMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, y = datasets.load_diabetes(return_X_y=True)
        cls.predictor = lgb.LGBMRegressor(
            n_estimators=10, max_depth=3, random_state=0, verbose=-1
        ).fit(cls.X, y)


if __name__ == "__main__":
    unittest.main()
