import os

import torch
from joblib import load

from ..fixed_formulation import FixedRegressionModel


def _make_diabetes_sequential():
    """Create a reproducible Sequential model for the diabetes dataset (10 features -> 1 output)."""
    torch.manual_seed(42)
    return torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )


class TestPytorchModel(FixedRegressionModel):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    basedir = os.path.join(os.path.dirname(__file__), "..", "predictors")

    def test_diabetes_pytorch(self):
        X = load(os.path.join(self.basedir, "examples_diabetes.joblib"))

        regressor = _make_diabetes_sequential()
        onecase = {"predictor": regressor, "nonconvex": 0}
        self.do_one_case(onecase, X, 5, "all")
        self.do_one_case(onecase, X, 6, "pairs")
