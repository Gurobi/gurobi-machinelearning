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

"""Module for formulating :external+sklearn:py:class:`sklearn.cross_decomposition.PLSRegression` in a gurobipy model."""

import numpy as np

from ..modeling import AbstractPredictorConstr
from .skgetter import SKgetter


def add_pls_regression_constr(
    gp_model, pls_regression, input_vars, output_vars=None, **kwargs
):
    """Formulate pls_regression in gp_model.

    The formulation predicts the values of output_vars using input_vars
    according to pls_regression.

    Parameters
    ----------
    gp_model : :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    pls_regression : :external+sklearn:py:class:`sklearn.cross_decomposition.PLSRegression`
     The linear regression to insert. It can be of any of the following types:
         * :external+sklearn:py:class:`sklearn.cross_decomposition.PLSRegression`
         * :external+sklearn:py:class:`sklearn.cross_decomposition.PLSCanonical`
     input_vars: mvar_array_like
         Decision variables used as input for random forest in model.
     output_vars: mvar_array_like, optional
         Decision variables used as output for random forest in model.

    Returns
    -------
    PLSRegressionConstr
        Object containing information about what was added to gp_model to
        formulate pls_regression.

    Notes
    -----
    |VariablesDimensionsWarn|
    """
    return PLSRegressionConstr(
        gp_model, pls_regression, input_vars, output_vars, **kwargs
    )


class PLSRegressionConstr(SKgetter, AbstractPredictorConstr):
    """Class to formulate a trained
    :external+sklearn:py:class:`sklearn.cross_decomposition.PLSRegression` in a
    gurobipy model.

    |ClassShort|
    """

    def __init__(
        self,
        gp_model,
        predictor,
        input_vars,
        output_vars=None,
        output_type="",
        **kwargs,
    ):
        self._output_shape = 1
        SKgetter.__init__(self, predictor, input_vars, output_type, **kwargs)
        AbstractPredictorConstr.__init__(
            self,
            gp_model,
            input_vars,
            output_vars,
            **kwargs,
        )

    def _add_regression_constr(self):
        """Add the prediction constraints to Gurobi."""
        x_mean = self.predictor._x_mean
        x_std = self.predictor._x_std
        coefs = self.predictor.coef_.T
        intercept = self.predictor.intercept_
        self.gp_model.addConstr(
            self.output
            == self.input @ (coefs / x_std[:, np.newaxis])
            - x_mean / x_std @ coefs
            + intercept,
            name="plsreg",
        )

    def _mip_model(self, **kwargs):
        """Add the prediction constraints to Gurobi."""
        self._add_regression_constr()
