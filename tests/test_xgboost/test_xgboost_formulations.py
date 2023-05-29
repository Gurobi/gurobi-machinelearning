import os

import numpy as np
import xgboost as xgb
from sklearn import datasets

from ..fixed_formulation import FixedRegressionModel


class TestXGBoosthModel(FixedRegressionModel):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    basedir = os.path.join(os.path.dirname(__file__), "..", "predictors")

    def test_diabetes_xgboost_pairs(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        xgb_reg = xgb.XGBRegressor(n_estimators=10, max_depth=4, max_leaf_nodes=10)
        xgb_reg.fit(X, y)
        one_case = {"predictor": xgb_reg.get_booster(), "nonconvex": 0}

        self.do_one_case(one_case, X, 6, "pairs", float_type=np.float32)

    def test_diabetes_xgboost_all(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        xgb_reg = xgb.XGBRegressor(n_estimators=10, max_depth=4, max_leaf_nodes=10)
        xgb_reg.fit(X, y)
        one_case = {"predictor": xgb_reg.get_booster(), "nonconvex": 0}

        self.do_one_case(one_case, X, 5, "all", epsilon=1e-5)
