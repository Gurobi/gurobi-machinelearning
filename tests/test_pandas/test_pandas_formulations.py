import unittest

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline

from gurobi_ml.sklearn.pipeline import PipelineConstr

from ..fixed_formulation import FixedRegressionModel
from ..test_sklearn.sklearn_cases import DiabetesCasesAsFrame, WageCase

VERBOSE = False


class TestSklearnPandasModel(FixedRegressionModel):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    def additional_test(self, predictor, pred_constr):
        if isinstance(predictor, Pipeline):
            self.assertIsInstance(pred_constr, PipelineConstr)
            self.assertEqual(len(predictor), len(pred_constr))
            for i in range(len(pred_constr)):
                predictor_name = type(predictor[i]).__name__
                if predictor_name in ["Lasso", "Ridge"]:
                    predictor_name = "LinearRegression"
                if predictor_name in ["PLSRegression", "PLSCanonical"]:
                    predictor_name = "PLSRegression"
                self.assertEqual(
                    predictor_name, type(pred_constr[i]).__name__[: -len("Constr")]
                )
                self.assertLessEqual(
                    np.max(pred_constr[i].get_error().astype(float)),
                    np.max(pred_constr.get_error()) + 1e-6,
                )

    def test_diabetes_sklearn_pandas(self):
        data = datasets.load_diabetes(as_frame=True)

        X = data["data"]
        cases = DiabetesCasesAsFrame()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            self.do_one_case(onecase, X, 5, "all", float_type=np.float32, epsilon=1e-5)
            self.do_one_case(onecase, X, 6, "pairs", float_type=np.float32)

    def test_wages(self):
        cases = WageCase()

        for regressor in cases:
            onecase = cases.get_case(regressor)
            X = onecase["data"]
            # onecase["nonconvex"] = 1
            self.do_one_case(
                onecase,
                X,
                6,
                numerical_features=cases.numerical_features,
                float_type=np.float32,
            )
            self.do_one_case(
                onecase,
                X,
                1,
                numerical_features=cases.numerical_features,
                float_type=np.float32,
            )


if __name__ == "__main__":
    VERBOSE = True
    unittest.main(verbosity=2)
