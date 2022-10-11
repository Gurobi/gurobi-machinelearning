# Copyright © 2022 Gurobi Optimization, LLC
from sklearn.utils.validation import check_is_fitted


class SKgetter:
    """Base class for all sklearn convertors"""

    def __init__(self, predictor):
        check_is_fitted(predictor)
        self.predictor = predictor

    def get_error(self):
        if self.has_solution():
            return self.predictor.predict(self.input.X) - self.output.X.T
        BaseException("No solution available")

    def get_error_proba(self):
        if self.has_solution():
            return self.predictor.predict_proba(self.input.X)[:, 1] - self.output.X.T
        BaseException("No solution available")
