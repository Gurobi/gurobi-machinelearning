# Copyright © 2022 Gurobi Optimization, LLC
from sklearn.utils.validation import check_is_fitted

from ..exceptions import NoSolution


class SKgetter:
    """Base class for all sklearn convertors

    Class used to implement some common functionalities

    Attributes
    ----------
    predictor
        Scikit-Learn predictor embedded into Gurobi model.
    """

    def __init__(self, predictor):
        check_is_fitted(predictor)
        self.predictor = predictor
        try:
            self.n_outputs_ = predictor.n_outputs_
        except AttributeError:
            self.n_outputs_ = 1

    def get_error(self):
        """Returns error in Gurobi's solution with respect to prediction from input

        Returns
        -------
        float
            Assuming that we have a solution for the input and output variables
            `x, y`. Returns the difference between `predict(x)` and
            `y`, where predict is the corresponding function for the Scikit-Learn
            object we are modeling.

        Raises
        ------
        NoSolution
            If the Gurobi model has no solution (either was not optimized or is infeasible).
        """
        if self._has_solution():
            return self.predictor.predict(self.input.X) - self.output.X.T
        raise NoSolution()

    def get_error_proba(self):
        """Same as get_error but using predict_proba

        Returns
        -------
        float
            Assuming that we have a solution for the input and output variables
            `x, y`. Returns the difference between `predict_proba(x)` and
            `y`, where predict is the corresponding function for the Scikit-Learn
            object we are modeling.

        Raises
        ------
        NoSolution
            If the Gurobi model has no solution (either was not optimized or is infeasible).
        """
        if self._has_solution():
            return self.predictor.predict_proba(self.input.X)[:, 1] - self.output.X.T
        raise NoSolution()
