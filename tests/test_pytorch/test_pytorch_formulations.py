import os

import torch
from joblib import load

from ..fixed_formulation import FixedRegressionModel


class TestPytorchModel(FixedRegressionModel):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    basedir = os.path.join(os.path.dirname(__file__), "..", "predictors")

    def test_diabetes_pytorch(self):
        X = load(os.path.join(self.basedir, "examples_diabetes.joblib"))

        filename = os.path.join(self.basedir, "diabetes__pytorch.pt")
        regressor = torch.load(filename, weights_only=False)
        onecase = {"predictor": regressor, "nonconvex": 0}
        self.do_one_case(onecase, X, 5, "all")
        self.do_one_case(onecase, X, 6, "pairs")
