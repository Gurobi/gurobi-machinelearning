import os

import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.pipeline import make_pipeline

from ..fixed_formulation import FixedRegressionModel


class TestXGBoosthModel(FixedRegressionModel):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    basedir = os.path.join(os.path.dirname(__file__), "..", "predictors")

    def test_diabetes_xgboost_pairs(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        xgb_reg = xgb.XGBRegressor(n_estimators=10, max_depth=4)
        xgb_reg.fit(X, y)
        one_case = {"predictor": xgb_reg.get_booster(), "nonconvex": 0}

        self.do_one_case(one_case, X, 6, "pairs", float_type=np.float32)

    def test_diabetes_xgboost_pipeline(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        xgb_reg = xgb.XGBRegressor(n_estimators=10, max_depth=4)
        pipeline = make_pipeline(xgb_reg)
        pipeline.fit(X, y)
        one_case = {"predictor": pipeline, "nonconvex": 0}

        self.do_one_case(one_case, X, 6, "pairs", float_type=np.float32)

    def test_diabetes_xgboost_all(self):
        data = datasets.load_diabetes()
        X = data["data"]
        y = data["target"]

        xgb_reg = xgb.XGBRegressor(n_estimators=10, max_depth=4)
        xgb_reg.fit(X, y)
        one_case = {"predictor": xgb_reg.get_booster(), "nonconvex": 0}

        self.do_one_case(one_case, X, 5, "all", epsilon=1e-5)

    @staticmethod
    def prepare_binary_iris():
        data = datasets.load_iris()
        X, y = data.data, data.target
        binary_mask = y != 2
        return X[binary_mask], y[binary_mask]

    @staticmethod
    def create_xgb_regressor(objective):
        return xgb.XGBRegressor(n_estimators=10, objective=objective, max_depth=4)

    def run_iris_test_case(self, predictor, X, method):
        one_case = {"predictor": predictor, "nonconvex": 0}
        self.do_one_case(one_case, X, 6, method, float_type=np.float32)

    def test_iris_xgboost_pipeline(self):
        X, y = self.prepare_binary_iris()
        pipeline = make_pipeline(self.create_xgb_regressor("binary:logistic"))
        pipeline.fit(X, y)
        self.run_iris_test_case(pipeline, X, "pairs")

    def test_iris_xgboost_pairs(self):
        X, y = self.prepare_binary_iris()
        xgb_reg = self.create_xgb_regressor("binary:logistic")
        xgb_reg.fit(X, y)
        self.run_iris_test_case(xgb_reg.get_booster(), X, "pairs")

    def test_iris_xgboost_all(self):
        X, y = self.prepare_binary_iris()
        xgb_reg = self.create_xgb_regressor("binary:logistic")
        xgb_reg.fit(X, y)
        self.run_iris_test_case(xgb_reg.get_booster(), X, "all")

    def test_iris_xgboost_reg_pipeline(self):
        X, y = self.prepare_binary_iris()
        pipeline = make_pipeline(self.create_xgb_regressor("reg:logistic"))
        pipeline.fit(X, y)
        self.run_iris_test_case(pipeline, X, "pairs")

    def test_iris_xgboost_reg_pairs(self):
        X, y = self.prepare_binary_iris()
        xgb_reg = self.create_xgb_regressor("reg:logistic")
        xgb_reg.fit(X, y)
        self.run_iris_test_case(xgb_reg.get_booster(), X, "pairs")

    def test_iris_xgboost_reg_all(self):
        X, y = self.prepare_binary_iris()
        xgb_reg = self.create_xgb_regressor("reg:logistic")
        xgb_reg.fit(X, y)
        self.run_iris_test_case(xgb_reg.get_booster(), X, "all")
