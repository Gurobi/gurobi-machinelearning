# Copyright © 2022 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Module for inserting simple Scikit-Learn regression models into a gurobipy model

All linear models should work:
   - :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
   - :external+sklearn:py:class:`sklearn.linear_model.Ridge`
   - :external+sklearn:py:class:`sklearn.linear_model.Lasso`

Also does :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression`
"""

import gurobipy as gp
import numpy as np

from ..exceptions import NoModel, ParameterError
from ..modeling import AbstractPredictorConstr
from .skgetter import SKgetter


def _name(index, name):
    index = f"{index}".replace(" ", "")
    return f"{name}[{index}]"


class BaseSKlearnRegressionConstr(SKgetter, AbstractPredictorConstr):
    """Predict a Gurobi variable using a Linear Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, predictor, input_vars, output_vars=None, output_type="", **kwargs):
        self.n_outputs_ = 1
        SKgetter.__init__(self, predictor, output_type, **kwargs)
        AbstractPredictorConstr.__init__(
            self,
            grbmodel,
            input_vars,
            output_vars,
            **kwargs,
        )

    def add_regression_constr(self):
        """Add the prediction constraints to Gurobi"""
        coefs = self.predictor.coef_.reshape(-1, 1)
        intercept = self.predictor.intercept_
        self.model.addConstr(self.output == self.input @ coefs + intercept, name="linreg")

    def print_stats(self, file=None):
        """Print statistics about submodel created"""
        super().print_stats(file)


class LinearRegressionConstr(BaseSKlearnRegressionConstr):
    """Predict a Gurobi variable using a linear regression that
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
    """Predict a Gurobi variable using a logistic regression that
    takes another Gurobi matrix variable as input.

    """

    def __init__(
        self,
        grbmodel,
        predictor,
        input_vars,
        output_vars=None,
        output_type="classification",
        epsilon=0.0,
        pwl_attributes=None,
        **kwargs,
    ):
        if len(predictor.classes_) > 2:
            raise NoModel("Logistic regression only supported for two classes")
        if pwl_attributes is None:
            self.attributes = self.default_pwl_attributes()
        else:
            self.attributes = pwl_attributes
        if output_type not in ("classification", "probability"):
            raise ParameterError("output_type should be either 'classification' or 'probability'")
        self.epsilon = epsilon

        BaseSKlearnRegressionConstr.__init__(
            self,
            grbmodel,
            predictor,
            input_vars,
            output_vars,
            output_type,
            **kwargs,
        )

    @staticmethod
    def default_pwl_attributes():
        """Default attributes for approximating the logistic function with Gurobi

        See `Gurobi's User Manual <https://www.gurobi.com/documentation/9.1/refman/general_constraint_attribu.html>`_
        for the meaning of the attributes.
        """
        return {"FuncPieces": -1, "FuncPieceLength": 0.01, "FuncPieceError": 0.01, "FuncPieceRatio": -1.0}

    def _mip_model(self):
        """Add the prediction constraints to Gurobi"""
        outputvars = self._output
        self._create_output_vars(self._input, name="affine_trans")
        affinevars = self._output
        self.add_regression_constr()
        if self.output_type == "classification":
            # For classification we need an extra binary variable
            self._create_output_vars(self._input, name="log_result")
            log_result = self._output
            self.model.addConstr(outputvars >= log_result - 0.5 + self.epsilon)
            self.model.addConstr(outputvars <= log_result + 0.5)
            outputvars.VType = gp.GRB.BINARY
            outputvars.LB = 0.0
            outputvars.UB = 1.0
        else:
            log_result = outputvars

        self._output = outputvars
        for index in np.ndindex(outputvars.shape):
            gc = self.model.addGenConstrLogistic(
                affinevars[index],
                log_result[index],
                name=_name(index, "logistic"),
            )
        numgc = self.model.NumGenConstrs
        self.model.update()
        for gc in self.model.getGenConstrs()[numgc:]:
            for attr, val in self.attributes.items():
                gc.setAttr(attr, val)
        self.model.update()


def add_linear_regression_constr(model, linear_regression, input_vars, output_vars=None, **kwargs):
    """Use `linear_regression` to predict the value of `output_vars` using `input_vars` in `model`

    Parameters
    ----------
    model: `gp.Model <https://www.gurobi.com/documentation/current/refman/py_model.html>`_
        The gurobipy model where the predictor should be inserted.
    linear_regression: : external+sklearn: py: class: `sklearn.linear_model.LinearRegression`
     The linear regression to insert. It can be of any of the following types:
         * : external+sklearn: py: class: `sklearn.linear_model.LinearRegression`
         * : external+sklearn: py: class: `sklearn.linear_model.Ridge`
         * : external+sklearn: py: class: `sklearn.linear_model.Lasso`
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
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars
    """
    return LinearRegressionConstr(model, linear_regression, input_vars, output_vars, **kwargs)


def add_logistic_regression_constr(
    model,
    logistic_regression,
    input_vars,
    output_vars=None,
    output_type="classification",
    epsilon=0.0,
    pwl_attributes=None,
    **kwargs,
):
    """Use `logistic_regression` to predict the value of `output_vars` using `input_vars` in `model`

    Parameters
    ----------

    model: `gp.Model <https://www.gurobi.com/documentation/current/refman/py_model.html>`_
        The gurobipy model where the predictor should be inserted.

    logistic_regression: :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression`
        The logistic regression to insert.

    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.

    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    output_type: {'classification', 'probability'}, default='classification'
        If the option chosen is 'classification' the output is the class label
        of either 0 or 1 given by the logistic regression.
        If the option 'probability' is chosen the output is the probabilty of the class 1.

    epsilon: float, default=0.0
        When the `output_type` is 'classification', this tolerance can be set
        to enforce that class 1 is chosen when the result of the logistic function is greater or
        equal to *0.5 + epsilon*.

        By default, with the value of *0.0*, if the result of the logistic function is
        very close to *0.5* (up to Gurobi tolerances) in the solution of the optimization model,
        the output of the regression can be either 0 or 1.
        The optimization model doesn't make a distinction between the two values.

        Setting *esilon* to a small value will remove this ambiguity on the output but may
        also make the model infeasible if the problem is very constrained:
        the open interval *(0.5, 0.5 + epsilon)* is excluded from the feasible set of the optimization
        problem.
    pwl_attributes: dict, optional
        Dictionary for non-default attributes for Gurobi to build the piecewise linear
        approximation of the logistic function.
        The default values for those attributes set in the package can be obtained
        with LogisticRegressionConstr.default_pwl_attributes().
        The dictionary keys should be the `attributes for modeling piece wise linear functions
        <https://www.gurobi.com/documentation/9.1/refman/general_constraint_attribu.html>`_
        and the values the corresponding value the users wants to pass to Gurobi.

    Returns
    -------
    LogisticRegressionConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Raises
    ------

    NoModel
        If the logistic regression is not a binary label regression

    ParameterError
        If the value of outut_type is set to a non-comforming value (see above).

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars
    """
    return LogisticRegressionConstr(
        model, logistic_regression, input_vars, output_vars, output_type, epsilon, pwl_attributes=pwl_attributes, **kwargs
    )
