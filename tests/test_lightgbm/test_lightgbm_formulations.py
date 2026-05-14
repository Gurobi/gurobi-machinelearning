import os

import lightgbm as lgb
import numpy as np
from sklearn import datasets
from sklearn.pipeline import make_pipeline

from ..fixed_formulation import FixedRegressionModel


class TestLGBMhModel(FixedRegressionModel):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    basedir = os.path.join(os.path.dirname(__file__), "..", "predictors")

    def test_diabetes_lightgbm_pairs(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        lgbm_reg = lgb.sklearn.LGBMRegressor(n_estimators=10, max_depth=4)
        lgbm_reg.fit(X, y)
        one_case = {"predictor": lgbm_reg, "nonconvex": 0}

        self.do_one_case(one_case, X, 6, "pairs", float_type=np.float32)

    def test_diabetes_lightgbm_pairs_pipeline(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        lgbm_reg = lgb.sklearn.LGBMRegressor(n_estimators=10, max_depth=4)
        pipeline = make_pipeline(lgbm_reg)
        pipeline.fit(X, y)
        one_case = {"predictor": pipeline, "nonconvex": 0}

        self.do_one_case(one_case, X, 6, "pairs", float_type=np.float32)

    def test_diabetes_lightgbm_all(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        lgbm_reg = lgb.sklearn.LGBMRegressor(n_estimators=10, max_depth=4)
        lgbm_reg.fit(X, y)
        one_case = {"predictor": lgbm_reg, "nonconvex": 0}

        self.do_one_case(one_case, X, 5, "all", epsilon=1e-5)
