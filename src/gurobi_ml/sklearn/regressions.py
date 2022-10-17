# Copyright © 2022 Gurobi Optimization, LLC
""" A set of classes for transforming Scikit-Learn regression objects
to constraints in Gurobi

What we have so far:
  - LinearRegressionConstr: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - LogisticRegressionConstr: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - MLSRegressionConstr: a neural network.
"""

import gurobipy as gp
import numpy as np

from ..modeling import AbstractPredictorConstr
from .skgetter import SKgetter


def _name(index, name):
    index = f"{index}".replace(" ", "")
    return f"{name}[{index}]"


class BaseSKlearnRegressionConstr(SKgetter, AbstractPredictorConstr):
    """Predict a Gurobi variable using a Linear Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, predictor, input_vars, output_vars=None, **kwargs):
        SKgetter.__init__(self, predictor)
        AbstractPredictorConstr.__init__(
            self,
            grbmodel,
            input_vars,
            output_vars,
            **kwargs,
        )

    def _create_output_vars(self, input_vars, name="linreg_out"):
        rval = self._model.addMVar((input_vars.shape[0], 1), lb=-gp.GRB.INFINITY, name=name)
        self._model.update()
        self._output = rval

    def add_regression_constr(self):
        """Add the prediction constraints to Gurobi"""
        coefs = self.predictor.coef_.T
        intercept = self.predictor.intercept_
        if self._output is None:
            self._create_output_vars(self._input)
        self.model.addConstr(self._output == self._input @ coefs + intercept, name="linreg")

    def print_stats(self, file=None):
        """Print statistics about submodel created"""
        super().print_stats(file)


class LinearRegressionConstr(BaseSKlearnRegressionConstr):
    """Predict a Gurobi variable using a Linear Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, predictor, input_vars, output_vars=None, **kwargs):
        BaseSKlearnRegressionConstr.__init__(
            self,
            grbmodel,
            predictor,
            input_vars,
            output_vars,
            **kwargs,
        )

    def _mip_model(self):
        """Add the prediction constraints to Gurobi"""
        self.add_regression_constr()


class LogisticRegressionConstr(BaseSKlearnRegressionConstr):
    """Predict a Gurobi variable using a Logistic Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, predictor, input_vars, output_vars=None, gc_attributes=None, **kwargs):
        if gc_attributes is None:
            self.attributes = {"FuncPieces": -1, "FuncPieceLength": 0.01, "FuncPieceError": 0.1, "FuncPieceRatio": -1.0}
        else:
            self.attributes = gc_attributes

        BaseSKlearnRegressionConstr.__init__(
            self,
            grbmodel,
            predictor,
            input_vars,
            output_vars,
            **kwargs,
        )

    def _mip_model(self):
        """Add the prediction constraints to Gurobi"""
        if self._output is None:
            self._create_output_vars(self._input, name="logreg")
        outputvars = self._output
        self._create_output_vars(self._input, name="affine_trans")
        affinevars = self._output
        self.add_regression_constr()
        self._output = outputvars
        for index in np.ndindex(outputvars.shape):
            gc = self.model.addGenConstrLogistic(
                affinevars[index],
                outputvars[index],
                name=_name(index, "logistic"),
            )
        numgc = self.model.NumGenConstrs
        self.model.update()
        for gc in self.model.getGenConstrs()[numgc:]:
            for attr, val in self.attributes.items():
                gc.setAttr(attr, val)


def add_linear_regression_constr(grbmodel, linear_regression, input_vars, output_vars=None, **kwargs):
    """Use `linear_regression` to predict the value of `output_vars` using `input_vars` in `model`

    Parameters
    ----------
    model: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
            The gurobipy model where the predictor should be inserted.
    linear_regression: :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
        The linear regression to insert.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    LinearRegressionConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Note
    ----
    See ... for acceptable values for input_vars and output_vars
    """
    return LinearRegressionConstr(grbmodel, linear_regression, input_vars, output_vars, **kwargs)


def add_logistic_regression_constr(grbmodel, logistic_regression, input_vars, output_vars=None, **kwargs):
    """Use `lositic_regression` to predict the value of `output_vars` using `input_vars` in `model`

    Parameters
    ----------
    model: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
            The gurobipy model where the predictor should be inserted.
    logistic_regression: :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression`
        The linear regression to insert.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    LinearRegressionConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Note
    ----
    See ... for acceptable values for input_vars and output_vars
    """
    return LogisticRegressionConstr(grbmodel, logistic_regression, input_vars, output_vars, **kwargs)
