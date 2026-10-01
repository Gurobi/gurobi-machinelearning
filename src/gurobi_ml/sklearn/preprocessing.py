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

"""Implementation for using sklearn preprocessing object in a
Gurobi model.
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from ..exceptions import ModelConfigurationError
from .skgetter import SKtransformer


def add_polynomial_features_constr(
    gp_model, polynomial_features, input_vars, output_vars=None, **kwargs
):
    """Formulate polynomial_features into gp_model.

    Note that this function creates the output variables from
    the input variables.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where polynomial features should be inserted.
    polynomial_features : :external+sklearn:py:class:`sklearn.preprocessing.PolynomialFeatures`
        The polynomial features to insert in gp_model.
    input_vars : mvar_array_like
        Decision variables used as input for polynomial features in gp_model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for polynomial features in gp_model.

    Returns
    -------
    PolynomialFeaturesConstr
        Object containing information about what was added to gp_model to insert the
        polynomial_features in it

    Warnings
    --------
    Only polynomial features of degree 2 are supported.
    """
    return PolynomialFeaturesConstr(
        gp_model, polynomial_features, input_vars, output_vars, **kwargs
    )


def add_standard_scaler_constr(
    gp_model, standard_scaler, input_vars, output_vars=None, **kwargs
):
    """Formulate standard_scaler into gp_model.

    Note that this function creates the output variables from
    the input variables.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the standard scaler should be inserted.
    standard_scaler : :external+sklearn:py:class:`sklearn.preprocessing.StandardScaler`
        The standard scaler to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for standard scaler in gp_model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for standard scaler in gp_model.

    Returns
    -------
    StandardScalerConstr
        Object containing information about what was added to gp_model to insert the
        standard_scaler in it
    """
    return StandardScalerConstr(
        gp_model, standard_scaler, input_vars, output_vars, **kwargs
    )


class StandardScalerConstr(SKtransformer):
    """Class to formulate a fitted
    :external+sklearn:py:class:`sklearn.preprocessing.StandardScaler` in a
    gurobipy model.

    Stores the changes to :external+gurobi:py:class:`Model` when formulating an instance into it.
    """

    def __init__(self, gp_model, scaler, input_vars, output_vars=None, **kwargs):
        self._default_name = "std_scaler"
        self._output_shape = scaler.n_features_in_
        super().__init__(gp_model, scaler, input_vars, output_vars, **kwargs)

    def _mip_model(self, **kwargs):
        """Do the transformation on x."""
        _input = self._input
        output = self._output

        scale = self.transformer.scale_
        mean = self.transformer.mean_

        if (
            kwargs.get("formulation") == "bigm"
            and kwargs.get("bigm") is None
            and self._output_created
        ):
            # Big-M formulations of the next steps need bounds on the output
            input_lb = _input.getAttr(GRB.Attr.LB)
            input_ub = _input.getAttr(GRB.Attr.UB)
            output.LB = np.where(
                input_lb <= -GRB.INFINITY, -GRB.INFINITY, (input_lb - mean) / scale
            )
            output.UB = np.where(
                input_ub >= GRB.INFINITY, GRB.INFINITY, (input_ub - mean) / scale
            )

        self.gp_model.addConstr(
            _input - output * scale == mean, name=self._name_var("s")
        )
        return self


class PolynomialFeaturesConstr(SKtransformer):
    """Class to formulate a trained
    :external+sklearn:py:class:`sklearn.preprocessing.PolynomialFeatures` in a
    gurobipy model.
    """

    def __init__(
        self, gp_model, polynomial_features, input_vars, output_vars=None, **kwargs
    ):
        if polynomial_features.degree > 2:
            raise ModelConfigurationError(
                polynomial_features, "Can only handle polynomials of degree <= 2"
            )
        self._default_name = "poly_feat"
        super().__init__(
            gp_model, polynomial_features, input_vars, output_vars, **kwargs
        )

    def _mip_model(self, **kwargs):
        """Do the transformation on x."""
        _input = self._input
        output = self._output

        n_examples, n_feat = _input.shape
        powers = self.transformer.powers_
        if powers.shape[0] != self.transformer.n_output_features_:
            raise RuntimeError(
                f"PolynomialFeatures internal inconsistency: powers.shape[0]={powers.shape[0]} "
                f"!= n_output_features_={self.transformer.n_output_features_}"
            )
        if powers.shape[1] != n_feat:
            raise RuntimeError(
                f"PolynomialFeatures internal inconsistency: powers.shape[1]={powers.shape[1]} "
                f"!= n_features={n_feat}"
            )

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

        if (
            kwargs.get("formulation") == "bigm"
            and kwargs.get("bigm") is None
            and self._output_created
        ):
            # Big-M formulations of the next steps need bounds on the output
            lb, ub = _monomials_bounds(
                _input.getAttr(GRB.Attr.LB), _input.getAttr(GRB.Attr.UB), powers
            )
            output.LB = lb
            output.UB = ub


def _monomials_bounds(input_lb, input_ub, powers):
    """Compute bounds on monomials of degree <= 2 by interval arithmetic.

    input_lb and input_ub are the bounds of the input variables (one row per
    example) and each row of powers gives the exponents of a monomial.
    """
    # Work with numpy infinities (Gurobi uses 1e100)
    input_lb = np.where(input_lb <= -GRB.INFINITY, -np.inf, input_lb)
    input_ub = np.where(input_ub >= GRB.INFINITY, np.inf, input_ub)

    def product(a_lb, a_ub, b_lb, b_ub):
        with np.errstate(invalid="ignore"):
            candidates = np.stack([a_lb * b_lb, a_lb * b_ub, a_ub * b_lb, a_ub * b_ub])
        # 0 * inf is 0 in interval arithmetic
        candidates = np.nan_to_num(candidates, nan=0.0, posinf=np.inf, neginf=-np.inf)
        return candidates.min(axis=0), candidates.max(axis=0)

    nex = input_lb.shape[0]
    lb = np.ones((nex, powers.shape[0]))
    ub = np.ones((nex, powers.shape[0]))
    for i, power in enumerate(powers):
        features = np.repeat(np.arange(len(power)), power)
        if len(features) == 1:
            lb[:, i] = input_lb[:, features[0]]
            ub[:, i] = input_ub[:, features[0]]
        elif len(features) == 2:
            f, g = features
            if f == g:
                # Square: 0 is the minimum if the interval contains it
                sq_lb, sq_ub = input_lb[:, f] ** 2, input_ub[:, f] ** 2
                ub[:, i] = np.maximum(sq_lb, sq_ub)
                lb[:, i] = np.where(
                    (input_lb[:, f] <= 0) & (input_ub[:, f] >= 0),
                    0.0,
                    np.minimum(sq_lb, sq_ub),
                )
            else:
                lb[:, i], ub[:, i] = product(
                    input_lb[:, f], input_ub[:, f], input_lb[:, g], input_ub[:, g]
                )
    return (
        np.where(np.isinf(lb), -GRB.INFINITY, lb),
        np.where(np.isinf(ub), GRB.INFINITY, ub),
    )


def sklearn_transformers():
    """Return dictionary of Scikit Learn preprocessing objects."""
    return {
        "StandardScaler": add_standard_scaler_constr,
        "PolynomialFeatures": add_polynomial_features_constr,
    }
