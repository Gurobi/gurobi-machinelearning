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

""" Implementation for using sklearn preprocessing object in a
Guobi model"""

import gurobipy as gp
import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..exceptions import NoModel, NoSolution
from ..modeling import AbstractPredictorConstr


def add_polynomial_features_constr(gp_model, polynomial_features, input_vars, **kwargs):
    """Embed polynomial_features into gp_model

    Note that this function creates the output variables from
    the input variables.

    Warning
    -------
    Only polynomial features of degree 2 are supported.

    Parameters
    ----------
    gp_model: :gurobipy:`model`
        The gurobipy model where polynomial features should be inserted.
    polynomial_features: :external+sklearn:py:class:`sklearn.preprocessing.PolynomialFeatures`
        The polynomial features to insert as predictor.
    input_vars: :gurobipy:`mvar` or :gurobipy:`var` array like
        Decision variables used as input for polynomial features in model.

    Returns
    -------
    sklearn.preprocessing.PolynomialFeaturesConstr
        Object containing information about what was added to gp_model to insert the
        polynomial_features in it

    """
    return PolynomialFeaturesConstr(gp_model, polynomial_features, input_vars, **kwargs)


def add_standard_scaler_constr(gp_model, standard_scaler, input_vars, **kwargs):
    """Embed standard_scaler into gp_model

    Note that this function creates the output variables from
    the input variables.

    Parameters
    ----------
    gp_model: :gurobipy:`model`
        The gurobipy model where the standard scaler should be inserted.
    standard_scaler: :external+sklearn:py:class:`sklearn.preprocessing.StandardScaler`
        The standard scaler to insert as predictor.
    input_vars: :gurobipy:`mvar` or :gurobipy:`var` array like
        Decision variables used as input for standard scaler in model.

    Returns
    -------
    sklearn.preprocessing.StandardScalerConstr
        Object containing information about what was added to gp_model to insert the
        standard_scaler in it

    """
    return StandardScalerConstr(gp_model, standard_scaler, input_vars, **kwargs)


class SKtransformer(AbstractPredictorConstr):
    def __init__(self, gp_model, transformer, input_vars, **kwargs):
        self.transformer = transformer
        if hasattr(transformer, "n_features_in_"):
            self._input_shape = transformer.n_features_in_
        if hasattr(transformer, "n_output_features_"):
            self._output_shape = transformer.n_output_features_
        check_is_fitted(transformer)
        super().__init__(gp_model, input_vars, **kwargs)

    def get_error(self):
        if self._has_solution():
            transformed = self.transformer.transform(self.input.X)
            if len(transformed.shape) == 1:
                transformed = transformed.reshape(-1, 1)
            return np.abs(transformed - self.output.X)
        raise NoSolution()


class StandardScalerConstr(SKtransformer):
    """Class to model trained :external+sklearn:py:class:`sklearn.preprocessing.StandardScaler` with gurobipy

    Stores the changes to :gurobipy:`model` when embedding an instance into it."""

    def __init__(self, gp_model, scaler, input_vars, **kwargs):
        self._default_name = "std_scaler"
        self._output_shape = scaler.n_features_in_
        super().__init__(gp_model, scaler, input_vars, **kwargs)

    def _mip_model(self, **kwargs):
        """Do the transformation on x"""
        _input = self._input
        output = self._output

        nfeat = _input.shape[1]
        scale = self.transformer.scale_
        mean = self.transformer.mean_

        self._gp_model.addConstr(_input - output * scale == mean, name="s")
        return self


class PolynomialFeaturesConstr(SKtransformer):
    """Class to model trained :external+sklearn:py:class:`sklearn.preprocessing.PolynomialFeatures` with gurobipy"""

    def __init__(self, gp_model, polynomial_features, input_vars, **kwargs):
        if polynomial_features.degree > 2:
            raise NoModel(
                polynomial_features, "Can only handle polynomials of degree < 2"
            )
        self._default_name = "poly_feat"
        super().__init__(gp_model, polynomial_features, input_vars, **kwargs)

    def _mip_model(self, **kwargs):
        """Do the transformation on x"""
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
                    output[k, i] == q_expr, name=f"polyfeat[{k},{i}]"
                )


def sklearn_transformers():
    return {
        "StandardScaler": add_standard_scaler_constr,
        "PolynomialFeatures": add_polynomial_features_constr,
    }


class ColumnTransformerConstr(SKtransformer):
    def __init__(self, gp_model, column_transformer, input_vars, **kwargs):
        self._default_name = "col_trans"
        super().__init__(gp_model, column_transformer, input_vars, **kwargs)

    def _create_output_vars(self, input_vars, **kwargs):
        out_shape = (input_vars.shape[0], self.transformer.n_output_features_)
        rval = self._gp_model.addMVar(out_shape, name="polyx", lb=-gp.GRB.INFINITY)
        self._gp_model.update()
        self._output = rval

    # For this class we need to reimplement submodel because we don't want
    # to transform input variables to Gurobi variable. We can't do it for categorical
    # The input should be unchanged.
    def _build_submodel(self, gp_model, *args, **kwargs):
        """Predict output from input using predictor or transformer"""
        self._mip_model(**kwargs)
        assert self._output is not None
        return self

    def _mip_model(self, **kwargs):
        """Do the transformation on x"""
        column_transform = self.transformer
        _input = self._input
        transformers = {k.lower(): v for k, v in sklearn_transformers().items()}
        transformed = []
        for name, trans, cols in column_transform.transformers_:
            if trans == "passthrough":
                transformed.append(_input.loc[:, cols])
            elif trans == "drop":
                pass
            else:
                data = _input.loc[:, cols]
                anyvar = any(
                    map(lambda i: isinstance(i, gp.Var), data.to_numpy().ravel())
                )
                if anyvar:
                    if name in transformers:
                        trans_constr = transformers[name](self._gp_model, trans, data)
                    transformed.append(trans_constr.output.tolist())
                else:
                    transformed.append(trans.transform(_input.loc[:, cols]))
        self._output = column_transform._hstack(transformed)


def add_column_transformer_constr(gp_model, column_transformer, input_vars, **kwargs):
    return ColumnTransformerConstr(gp_model, column_transformer, input_vars, **kwargs)
