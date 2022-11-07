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

""" Module for embedding :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression` into a
:gurobipy:`model`
"""

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
    """Embed logistic_regression into gp_model

    Predict the values of output_vars using input_vars

    Parameters
    ----------

    gp_model: :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    logistic_regression: :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression`
        The logistic regression to insert.
    input_vars: :gurobipy:`mvar` or :gurobipy:`var` array like
        Decision variables used as input for logistic regression in model.
    output_vars: :gurobipy:`mvar` or :gurobipy:`var` array like, optional
        Decision variables used as output for logistic regression in model.

    output_type: {'classification', 'probability_1'}, default='classification'
        If the option chosen is 'classification' the output is the class label
        of either 0 or 1 given by the logistic regression.
        If the option 'probability_1' is chosen the output is the probability of the class 1.

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
        Object containing information about what was added to gp_model to embed the
        predictor into it

    Raises
    ------

    NoModel
        If the logistic regression is not a binary label regression

    ParameterError
        If the value of output_type is set to a non-conforming value (see above).

    Note
    ----
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
    """Class to model trained :external+sklearn:py:class:`sklearn.linear_model.LogisticRegression` with gurobipy

    Stores the changes to :gurobipy:`model` when embedding an instance into it."""

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
            raise NoModel(predictor, "Logistic regression only supported for two classes")
        if pwl_attributes is None:
            self.attributes = self.default_pwl_attributes()
        else:
            self.attributes = pwl_attributes
        if output_type not in ("classification", "probability_1"):
            raise ParameterError("output_type should be either 'classification' or 'probability_1'")
        self.epsilon = epsilon
        self._default_name = "log_reg"
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
    def default_pwl_attributes():
        """Default attributes for approximating the logistic function with Gurobi

        See `Gurobi's User Manual <https://www.gurobi.com/documentation/9.1/refman/general_constraint_attribu.html>`_
        for the meaning of the attributes.
        """
        return {
            "FuncPieces": -1,
            "FuncPieceLength": 0.01,
            "FuncPieceError": 0.01,
            "FuncPieceRatio": -1.0,
        }

    def _mip_model(self, **kwargs):
        """Add the prediction constraints to Gurobi"""
        outputvars = self._output
        self._create_output_vars(self._input, name="affine_trans")
        affinevars = self._output
        self.add_regression_constr()
        if self.output_type == "classification":
            # For classification we need an extra binary variable
            self._create_output_vars(self._input, name="log_result")
            log_result = self._output
            self.gp_model.addConstr(outputvars >= log_result - 0.5 + self.epsilon)
            self.gp_model.addConstr(outputvars <= log_result + 0.5)
            outputvars.VType = gp.GRB.BINARY
            outputvars.LB = 0.0
            outputvars.UB = 1.0
        else:
            log_result = outputvars

        self._output = outputvars
        for index in np.ndindex(outputvars.shape):
            gc = self.gp_model.addGenConstrLogistic(
                affinevars[index],
                log_result[index],
                name=self._indexed_name(index, "logistic"),
            )
        num_gc = self.gp_model.NumGenConstrs
        self.gp_model.update()
        for gc in self.gp_model.getGenConstrs()[num_gc:]:
            for attr, val in self.attributes.items():
                gc.setAttr(attr, val)
        self.gp_model.update()
