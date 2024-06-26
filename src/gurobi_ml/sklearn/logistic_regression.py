# Copyright © 2023 Gurobi Optimization, LLC
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
:gurobipy:`model`.
"""

import math
import warnings

import gurobipy as gp
import numpy as np

from ..exceptions import NoModel, ParameterError
from .base_regressions import BaseSKlearnRegressionConstr


def add_logistic_regression_constr(
    gp_model,
    logistic_regression,
    input_vars,
    output_vars=None,
    output_type="classification",
    epsilon=0.0,
    pwl_attributes=None,
    **kwargs,
):
    """Formulate logistic_regression in gp_model.

    The formulation predicts the values of output_vars using input_vars according to
    logistic_regression.

    When the desired output type is classification the result can be achieved without
    appealing to the logistic function, and consequently non-linearity can be avoided.
    When the desired output type is regression then the (non-linear) logistic function
    is required. For users of Gurobi ≥ 11.0, the attribute FuncNonlinear is set to 1 to
    deal directly with the logistic function in an algorithmic fashion.

    For older versions, Gurobi makes a piecewise linear approximation of the logistic
    function.
    The quality of the approximation can be controlled with the parameter
    pwl_attributes. By default, it is parameterized so that the maximal error of the
    approximation is `1e-2`.

    See our :ref:`Users Guide <Logistic Regression>` for
    details on the mip formulation used.

    Parameters
    ----------

    gp_model : :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    logistic_regression : :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression`
        The logistic regression to insert.
    input_vars : mvar_array_like
        Decision variables used as input for logistic regression in model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for logistic regression in model.

    output_type : {'classification', 'probability_1'}, default='classification'
        If the option chosen is 'classification' the output is the class label
        of either 0 or 1 given by the logistic regression. If the option
        'probability_1' is chosen the output is the probability of the class 1.

    epsilon : float, default=0.0
        When the `output_type` is 'classification', this tolerance can be set
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
        If the value of output_type is set to a non-conforming value (see above).

    Notes
    -----
    |VariablesDimensionsWarn|
    """
    return LogisticRegressionConstr(
        gp_model,
        logistic_regression,
        input_vars,
        output_vars,
        output_type,
        epsilon,
        pwl_attributes=pwl_attributes,
        **kwargs,
    )


class LogisticRegressionConstr(BaseSKlearnRegressionConstr):
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
        output_type="classification",
        epsilon=0.0,
        pwl_attributes=None,
        **kwargs,
    ):
        if len(predictor.classes_) > 2:
            raise NoModel(
                predictor, "Logistic regression only supported for two classes"
            )
        if output_type not in ("classification", "probability_1"):
            raise ParameterError(
                "output_type should be either 'classification' or 'probability_1'"
            )
        if output_type == "classification" and pwl_attributes is not None:
            message = """
pwl_attributes are not required for classification.  The problem is
formulated without requiring the non-linear logistic function."""
            warnings.warn(message)
        elif output_type != "classification":
            self.attributes = (
                self.default_pwl_attributes()
                if pwl_attributes is None
                else pwl_attributes
            )

        self.epsilon = epsilon
        self._default_name = "log_reg"
        self.affinevars = None
        BaseSKlearnRegressionConstr.__init__(
            self,
            gp_model,
            predictor,
            input_vars,
            output_vars,
            output_type,
            **kwargs,
        )

    @staticmethod
    def default_pwl_attributes() -> dict:
        """Default attributes for approximating the logistic function with Gurobi.

        See `Gurobi's User Manual
        <https://www.gurobi.com/documentation/current/refman/general_constraint_attribu.html>`_
        for the meaning of the attributes.
        """
        if gp.gurobi.version()[0] < 11:
            message = """
Gurobi ≥ 11 can deal directly with nonlinear functions with 'FuncNonlinear'.
Upgrading to version 11 is recommended when using logistic regressions."""
            warnings.warn(message)
            return {
                "FuncPieces": -1,
                "FuncPieceLength": 0.01,
                "FuncPieceError": 0.01,
                "FuncPieceRatio": -1.0,
            }
        return {"FuncNonlinear": 1}

    def _addGenConstrIndicatorMvarV10(self, binvar, binval, lhs, sense, rhs, name):
        """This function is to work around the lack of MVar compatibility in
        Gurobi v10 indicator constraints.  Note, it is not as flexible as Model.addGenConstrIndicator
        in V11+.  If support for v10 is dropped this function can be removed.

        Parameters
        ----------
        binvar : MVar
        binval : {0,1}
        lhs : MVar or MLinExpr
        sense : (char)
            Options are gp.GRB.LESS_EQUAL, gp.GRB.EQUAL, or gp.GRB.GREATER_EQUAL
        rhs : scalar
        name : string
        """
        assert binvar.shape == lhs.shape
        total_constraints = np.prod(binvar.shape)
        binvar = binvar.reshape(total_constraints).tolist()
        lhs = lhs.reshape(total_constraints).tolist()
        for index in range(total_constraints):
            self.gp_model.addGenConstrIndicator(
                binvar[index],
                binval,
                lhs[index],
                sense,
                rhs,
                name=self._indexed_name(index, name),
            )

    def _mip_model(self, **kwargs):
        """Add the prediction constraints to Gurobi."""
        affinevars = self.gp_model.addMVar(
            self.output.shape, lb=-gp.GRB.INFINITY, name="affine_trans"
        )
        self._add_regression_constr(output=affinevars)

        if self.output_type == "classification":
            # For classification we need an extra binary variable
            bin_output = self.gp_model.addMVar(
                self.output.shape, vtype=gp.GRB.BINARY, name="bin_output"
            )

            # Workaround for MVars in indicator constraints for v10.
            addGenConstrIndicator = (
                self.gp_model.addGenConstrIndicator
                if gp.gurobi.version()[0] >= 11
                else self._addGenConstrIndicatorMvarV10
            )

            # The original epsilon is with respect to the range of the logistic function.
            # We must translate this to the domain of the logistic function.
            affine_trans_epsilon = -math.log(1 / (0.5 + self.epsilon) - 1)

            # For classification it is enough to test result of affine transformation
            # and avoid adding logistic curve constraint.  See GH316.
            addGenConstrIndicator(
                bin_output,
                1,
                affinevars,
                gp.GRB.GREATER_EQUAL,
                affine_trans_epsilon,
                "indicator_affinevars_pos",
            )
            addGenConstrIndicator(
                bin_output,
                0,
                affinevars,
                gp.GRB.LESS_EQUAL,
                0,
                "indicator_affinevars_neg",
            )
            self.gp_model.addConstr(bin_output == self.output)
        else:
            log_result = self.output
            for index in np.ndindex(self.output.shape):
                self.gp_model.addGenConstrLogistic(
                    affinevars[index],
                    log_result[index],
                    name=self._indexed_name(index, "logistic"),
                )
            num_gc = self.gp_model.NumGenConstrs
            self.gp_model.update()
            for gen_constr in self.gp_model.getGenConstrs()[num_gc:]:
                for attr, val in self.attributes.items():
                    gen_constr.setAttr(attr, val)
        self.gp_model.update()

    @property
    def affine_transformation_variables(self) -> gp.MVar:
        """Variables that store the result of the affine transformation from the regression coefficient.
        (intermediate result before applying the logistic function).
        """
        return self.affinevars
