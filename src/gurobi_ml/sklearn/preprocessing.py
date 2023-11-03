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

"""Implementation for using sklearn preprocessing object in a
Guobi model.
"""

import gurobipy as gp

from ..exceptions import NoModel
from .skgetter import SKtransformer


def add_polynomial_features_constr(gp_model, polynomial_features, input_vars, **kwargs):
    """Formulate polynomial_features into gp_model.

    Note that this function creates the output variables from
    the input variables.

    Parameters
    ----------
    gp_model : :gurobipy:`model`
        The gurobipy model where polynomial features should be inserted.
    polynomial_features : :external+sklearn:py:class:`sklearn.preprocessing.PolynomialFeatures`
        The polynomial features to insert in gp_model.
    input_vars : mvar_array_like
        Decision variables used as input for polynomial features in model.

    Returns
    -------
    sklearn.preprocessing.PolynomialFeaturesConstr
        Object containing information about what was added to gp_model to insert the
        polynomial_features in it

    Warnings
    --------
    Only polynomial features of degree 2 are supported.
    """
    return PolynomialFeaturesConstr(gp_model, polynomial_features, input_vars, **kwargs)


def add_standard_scaler_constr(gp_model, standard_scaler, input_vars, **kwargs):
    """Formulate standard_scaler into gp_model.

    Note that this function creates the output variables from
    the input variables.

    Parameters
    ----------
    gp_model : :gurobipy:`model`
        The gurobipy model where the standard scaler should be inserted.
    standard_scaler : :external+sklearn:py:class:`sklearn.preprocessing.StandardScaler`
        The standard scaler to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for standard scaler in model.

    Returns
    -------
    sklearn.preprocessing.StandardScalerConstr
        Object containing information about what was added to gp_model to insert the
        standard_scaler in it
    """
    return StandardScalerConstr(gp_model, standard_scaler, input_vars, **kwargs)


class StandardScalerConstr(SKtransformer):
    """Class to formulate a fitted
    :external+sklearn:py:class:`sklearn.preprocessing.StandardScaler` in a
    gurobipy model.

    Stores the changes to :gurobipy:`model` when formulating an instance into it.
    """

    def __init__(self, gp_model, scaler, input_vars, **kwargs):
        self._default_name = "std_scaler"
        self._output_shape = scaler.n_features_in_
        super().__init__(gp_model, scaler, input_vars, **kwargs)

    def _mip_model(self, **kwargs):
        """Do the transformation on x."""
        _input = self._input
        output = self._output

        scale = self.transformer.scale_
        mean = self.transformer.mean_

        self.gp_model.addConstr(
            _input - output * scale == mean, name=self._name_var("s")
        )
        return self


class PolynomialFeaturesConstr(SKtransformer):
    """Class to formulate a trained
    :external+sklearn:py:class:`sklearn.preprocessing.PolynomialFeatures` in a
    gurobipy model.
    """

    def __init__(self, gp_model, polynomial_features, input_vars, **kwargs):
        if polynomial_features.degree > 2:
            raise NoModel(
                polynomial_features, "Can only handle polynomials of degree < 2"
            )
        self._default_name = "poly_feat"
        super().__init__(gp_model, polynomial_features, input_vars, **kwargs)

    def _mip_model(self, **kwargs):
        """Do the transformation on x."""
        _input = self._input
        output = self._output

        n_examples, n_feat = _input.shape
        powers = self.transformer.powers_
        assert powers.shape[0] == self.transformer.n_output_features_
        assert powers.shape[1] == n_feat

        for k in range(n_examples):
            for i, power in enumerate(powers):
                q_expr = gp.QuadExpr()
                q_expr += 1.0
                for j, feat in enumerate(_input[k, :]):
                    if power[j] == 2:
                        q_expr *= feat.item()
                        q_expr *= feat.item()
                    elif power[j] == 1:
                        q_expr *= feat.item()
                self.gp_model.addConstr(
                    output[k, i] == q_expr, name=self._indexed_name((k, i), "polyfeat")
                )


def sklearn_transformers():
    """Return dictionary of Scikit Learn preprocessing objects."""
    return {
        "StandardScaler": add_standard_scaler_constr,
        "PolynomialFeatures": add_polynomial_features_constr,
    }
