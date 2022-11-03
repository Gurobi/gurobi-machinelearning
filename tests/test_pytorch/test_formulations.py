import os

import torch
from sklearn import datasets

from ..fixed_formulation import FixedRegressionModel


class TestPytorchModel(FixedRegressionModel):
    """Test that if we fix the input of the predictor the feasible solution from
    Gurobi is identical to what the predict function would return."""

    def additional_test(self, predictor, pred_constr):
        # No addtional test
        pass

    def test_diabetes_pytorch(self):
        data = datasets.load_diabetes()

        X = data["data"]

        filename = os.path.join(
            os.path.dirname(__file__), "..", "predictors", "diabetes__pytorch.pt"
        )
        regressor = torch.load(filename)
        onecase = {"predictor": regressor, "nonconvex": 0}
        self.do_one_case(onecase, X, 5, "all")
        self.do_one_case(onecase, X, 6, "pairs")
