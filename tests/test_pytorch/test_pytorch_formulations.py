import os

import torch
from joblib import load

from ..fixed_formulation import FixedRegressionModel


class TestPytorchModel(FixedRegressionModel):
    """Test the default MIP (bigm) ReLU formulation using a pre-saved model.

    Unlike test_pytorch_activations.py (which always uses NonConvex=2), this
    test uses nonconvex=0 — exercising the LP/MIP embedding path for ReLU
    that does not require the NonConvex solver parameter.
    """

    basedir = os.path.join(os.path.dirname(__file__), "..", "predictors")

    def test_diabetes_pytorch(self):
        X = load(os.path.join(self.basedir, "examples_diabetes.joblib"))

        filename = os.path.join(self.basedir, "diabetes__pytorch.pt")
        regressor = torch.load(filename, weights_only=False)
        onecase = {"predictor": regressor, "nonconvex": 0}
        self.do_one_case(onecase, X, 5, "all")
        self.do_one_case(onecase, X, 6, "pairs")
