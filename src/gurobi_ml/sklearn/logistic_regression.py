# Copyright © 2023-2026 Gurobi Optimization, LLC
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

"""Module for formulating a
:external+sklearn:py:class:`sklearn.linear_model.LogisticRegression` in a
:external+gurobi:py:class:`Model`.
"""

import warnings

import gurobipy as gp

from ..modeling.base_predictor_constr import AbstractPredictorConstr
from ..modeling.softmax import logistic, softmax

try:
    pass

    _HAS_NL_EXPR = True
    _HAS_NL = True
except ImportError:
    if gp.gurobi.version()[0] < 11:
        _HAS_NL = False
    else:
        _HAS_NL = True
    _HAS_NL_EXPR = False

from .skgetter import SKClassifier


def add_logistic_regression_constr(
    gp_model,
    logistic_regression,
    input_vars,
    output_vars=None,
    predict_function="predict_proba",
    epsilon=0.0,
    pwl_attributes=None,
    **kwargs,
):
    """Formulate logistic_regression in gp_model.

    The formulation predicts the values of output_vars using input_vars according to
    logistic_regression.

    When the desired output type is classification a mixed integer linear formulation
    using indicator constraints is used. When the desired output type is regression
    then the (non-linear) logistic function is required. For users of Gurobi ≥ 11.0,
    the attribute FuncNonlinear is set to 1 to deal directly with the logistic
    function in an algorithmic fashion.

    For older versions, Gurobi makes a piecewise linear approximation of the logistic
    function.
    The quality of the approximation can be controlled with the parameter
    pwl_attributes. By default, it is parametrized so that the maximal error of the
    approximation is `1e-2`.

    See our :ref:`Users Guide <Logistic Regression>` for
    details on the mip formulation used.

    Parameters
    ----------

    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    logistic_regression : :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression`
        The logistic regression to insert.
    input_vars : mvar_array_like
        Decision variables used as input for logistic regression in model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for logistic regression in model.

    predict_function: {'predict', 'predict_proba'}, default='predict'
        If the option chosen is 'predict' the output is the class label
        of either 0 or 1 given by the logistic regression. If the option
        'predict_proba' is chosen the output is the probability of each class.

    epsilon : float, default=0.0
        When the `predict_function` is 'predict', this tolerance can be set
        to enforce that class 1 is chosen when the result of the logistic
        function is greater or equal to *0.5 + epsilon*.

        By default, with the value of *0.0*, if the result of the logistic
        function is very close to *0.5* (up to Gurobi tolerances) in the
        solution of the optimization model, the output of the regression can be
        either 0 or 1. The optimization model doesn't make a distinction
        between the two values.

        Setting *esilon* to a small value will remove this ambiguity on the
        output but may also make the model infeasible if the problem is very
        constrained: the open interval *(0.5, 0.5 + epsilon)* is excluded from
        the feasible set of the optimization problem.

    pwl_attributes : dict, optional
        Dictionary for non-default attributes for Gurobi to build the piecewise
        linear approximation of the logistic function. This is only relevent when
        the output type is regression, not classification. The default values for
        those attributes set in the package can be obtained with
        LogisticRegressionConstr.default_pwl_attributes(). The dictionary keys
        should be the `attributes for modeling piece wise linear functions
        <https://www.gurobi.com/documentation/9.1/refman/general_constraint_attribu.html>`_
        and the values the corresponding value the users wants to pass to
        Gurobi.

    Returns
    -------
    LogisticRegressionConstr
        Object containing information about what was added to gp_model to formulate
        logistic_regression.

    Raises
    ------

    NoModel
        If the logistic regression is not a binary label regression

    ParameterError
        If the value of predict_function is set to a non-conforming value (see above).

    Notes
    -----
    |VariablesDimensionsWarn|
    """
    return LogisticRegressionConstr(
        gp_model,
        logistic_regression,
        input_vars,
        output_vars,
        predict_function,
        epsilon,
        pwl_attributes=pwl_attributes,
        **kwargs,
    )


class LogisticRegressionConstr(SKClassifier, AbstractPredictorConstr):
    """Class to formulate a trained
    :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression` in a gurobipy model.

    |ClassShort|
    """

    def __init__(
        self,
        gp_model,
        predictor,
        input_vars,
        output_vars=None,
        predict_function="predict_proba",
        epsilon=0.0,
        pwl_attributes=None,
        **kwargs,
    ):
        if predict_function not in ("predict_proba", "decision_function"):
            raise ValueError(
                "predict_function should be either 'predict_proba' or 'decision_function'"
            )
        if predict_function != "predict_proba" and pwl_attributes is not None:
            message = """
pwl_attributes are not required for classification.  The problem is
formulated without requiring the non-linear logistic function."""
            warnings.warn(message)
        elif predict_function == "predict_proba":
            self.attributes = (
                self.default_pwl_attributes()
                if pwl_attributes is None
                else pwl_attributes
            )

        self.epsilon = epsilon
        self._default_name = "log_reg"
        self.linear_predictor = None
        SKClassifier.__init__(self, predictor, input_vars, predict_function)
        if self._output_shape == 2 and predict_function == "decision_function":
            self._output_shape = 1
        AbstractPredictorConstr.__init__(
            self,
            gp_model,
            input_vars,
            output_vars,
            **kwargs,
        )

    @staticmethod
    def default_pwl_attributes() -> dict:
        """Default attributes for approximating the logistic function with Gurobi.

        See `Gurobi's User Manual
        <https://www.gurobi.com/documentation/current/refman/general_constraint_attribu.html>`_
        for the meaning of the attributes.
        """
        if not _HAS_NL:
            message = """
Gurobi ≥ 11 can deal directly with nonlinear functions with 'FuncNonlinear'.
Upgrading to version 11 is recommended when using logistic regressions."""
            warnings.warn(message)
            return {
                "FuncPieces": -1,
                "FuncPieceLength": 0.01,
                "FuncPieceError": 0.01,
                "FuncPieceRatio": -1.0,
                "FuncNonlinear": 0,
            }
        if not _HAS_NL_EXPR:
            message = """
Gurobi ≥ 12 can deal directly with nonlinear expressions.
Upgrading to version 12 is recommended when using logistic regressions."""
            warnings.warn(message)
            return {"FuncNonlinear": 1}
        return {}

    def _two_classes_model(self, **kwargs):
        """Add the prediction constraints to Gurobi."""
        coefs = self.predictor.coef_
        intercept = self.predictor.intercept_

        linreg = self.input @ coefs.T + intercept

        self.linear_predictor = linreg

        if self.predict_function == "predict_proba":
            self.gp_model.addConstr(self.output.sum(axis=1) == 1)
            logistic(self, linreg)
        else:
            self.gp_model.addConstr(self.output[:, 0] == linreg[:, 0])

        self.gp_model.update()

    def _multi_class_model(self, **kwargs):
        """Add the prediction constraints to Gurobi."""
        coefs = self.predictor.coef_
        intercept = self.predictor.intercept_

        linreg = self.input @ coefs.T + intercept

        if self.predict_function == "predict_proba":
            softmax(self, linreg, **kwargs)
        else:
            self.gp_model.addConstr(self.output == linreg)

    def _mip_model(self, **kwargs):
        if self._output_shape > 2:
            return self._multi_class_model(**kwargs)
        return self._two_classes_model(**kwargs)
