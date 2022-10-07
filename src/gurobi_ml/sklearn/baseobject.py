from sklearn.utils.validation import check_is_fitted


class SKgetter:
    """Base class for all sklearn convertors"""

    def __init__(self, predictor):
        check_is_fitted(predictor)
        self.predictor = predictor

    def get_error(self):
        if self.sol_available():
            return self.predictor.predict(self.input.X) - self.output.X.T
        return 0.0
