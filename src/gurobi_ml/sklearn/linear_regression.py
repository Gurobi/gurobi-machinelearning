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

"""Module for formulating ordinary regression models into a
:gurobipy:`model`.

The following linear models are tested and should work:
   - :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
   - :external+sklearn:py:class:`sklearn.linear_model.Ridge`
   - :external+sklearn:py:class:`sklearn.linear_model.Lasso`
"""

from .base_regressions import BaseSKlearnRegressionConstr


def add_linear_regression_constr(
    gp_model, linear_regression, input_vars, output_vars=None, **kwargs
):
    """Formulate linear_regression in gp_model.

    The formulation predicts the values of output_vars using input_vars according to
    linear_regression. See our :ref:`Users Guide <Linear Regression>` for details on the
    mip formulation used.

    Parameters
    ----------
    gp_model : :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    linear_regression : :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
     The linear regression to insert. It can be of any of the following types:
         * :external+sklearn:py:class:`sklearn.linear_model.LinearRegression`
         * :external+sklearn:py:class:`sklearn.linear_model.Ridge`
         * :external+sklearn:py:class:`sklearn.linear_model.Lasso`
     input_vars: mvar_array_like
         Decision variables used as input for random forest in model.
     output_vars: mvar_array_like, optional
         Decision variables used as output for random forest in model.

    Returns
    -------
    LinearRegressionConstr
        Object containing information about what was added to gp_model to formulate
        linear_regression.

    Notes
    -----
    |VariablesDimensionsWarn|
    """
    return LinearRegressionConstr(
        gp_model, linear_regression, input_vars, output_vars, **kwargs
    )


class LinearRegressionConstr(BaseSKlearnRegressionConstr):
    """Class to formulate a trained
    :external+sklearn:py:class:`sklearn.linear_model.LinearRegression` in a gurobipy model.

    |ClassShort|.
    """

    def __init__(self, gp_model, predictor, input_vars, output_vars=None, **kwargs):
        self._default_name = "lin_reg"
        BaseSKlearnRegressionConstr.__init__(
            self,
            gp_model,
            predictor,
            input_vars,
            output_vars,
            **kwargs,
        )

    def _mip_model(self, **kwargs):
        """Add the prediction constraints to Gurobi."""
        self._add_regression_constr()
