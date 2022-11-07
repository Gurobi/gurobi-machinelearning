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

""" Module for inserting ordinary Scikit-Learn regression models into a :gurobipy:`model`

The following linear models are tested and should work:
   - :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
   - :external+sklearn:py:class:`sklearn.linear_model.Ridge`
   - :external+sklearn:py:class:`sklearn.linear_model.Lasso`
"""

from .base_regressions import BaseSKlearnRegressionConstr


def add_linear_regression_constr(
    gp_model, linear_regression, input_vars, output_vars=None, **kwargs
):
    """Embed linear_regression in gp_model to predict the values of output_vars using input_vars

    Parameters
    ----------
    gp_model: :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    linear_regression: :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
     The linear regression to insert. It can be of any of the following types:
         * :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
         * :external+sklearn:py:class:`sklearn.linear_model.Ridge`
         * :external+sklearn:py:class:`sklearn.linear_model.Lasso`
     input_vars: :gurobipy:`mvar` or :gurobipy:`var` array like
         Decision variables used as input for random forest in model.
     output_vars: :gurobipy:`mvar` or :gurobipy:`var` array like, optional
         Decision variables used as output for random forest in model.

    Returns
    -------
    LinearRegressionConstr
        Object containing information about what was added to gp_model to embed the
        predictor into it

    Note
    ----
    |VariablesDimensionsWarn|
    """
    return LinearRegressionConstr(gp_model, linear_regression, input_vars, output_vars, **kwargs)


class LinearRegressionConstr(BaseSKlearnRegressionConstr):
    """Class to model trained :external+sklearn:py:class:`sklearn.linear_model.LinearRegression` with gurobipy

    Stores the changes to :gurobipy:`model` when embedding an instance into it."""

    def __init__(self, gp_model, predictor, input_vars, output_vars=None, **kwargs):
        self._default_name = ("lin_reg",)
        BaseSKlearnRegressionConstr.__init__(
            self,
            gp_model,
            predictor,
            input_vars,
            output_vars,
            **kwargs,
        )

    def _mip_model(self, **kwargs):
        """Add the prediction constraints to Gurobi"""
        self.add_regression_constr()
