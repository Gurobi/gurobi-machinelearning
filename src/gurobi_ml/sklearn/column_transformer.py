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

"""Module for formulating a :external+sklearn:py:class:`sklearn.compose.ColumnTransformer` in a :gurobipy:`gurobipy model <Model>`."""

import gurobipy as gp

from ..exceptions import NoModel
from .preprocessing import sklearn_transformers
from .skgetter import SKtransformer


class ColumnTransformerConstr(SKtransformer):
    """Class to formulate a fitted :external+sklearn:py:class:`sklearn.compose.ColumnTransformer` in a gurobipy model.

    Notes
    -----
    This object differs from all the other object in the Gurobi Machine Learning package in
    that it may not be possible to write all of its input with Gurobi variables. Specifically
    some input may consist of categorical features to be encoded using the column transformer.

    Then such input cannot be directly represented in Gurobi but the result of the encoding may be represented.
    If the categorical features are fixed for the optimization model, we can use it so it is allowed by
    the ColumnTransormerConstr class.

    The rule we use to apply the ColumnTransformer to the input is that if the set of columns to which a preprocessing
    transformation is constant in the input we use directly the scikit learn preprocessing object. It at least one of the columns
    is made of gurobipy variables, we use the gurobi-ml object (if it exists).
    """

    def __init__(self, gp_model, column_transformer, input_vars, **kwargs):
        self._default_name = "col_trans"
        super().__init__(gp_model, column_transformer, input_vars, **kwargs)

    # For this class we need to reimplement submodel because we don't want
    # to transform input variables to Gurobi variable. We can't do it for categorical
    # The input should be unchanged.
    def _build_submodel(self, gp_model, *args, **kwargs):
        """Predict output from input using predictor or transformer."""
        _input = self.input
        if hasattr(self._input, "columns"):
            self._input_columns = _input.columns
        if hasattr(self._input, "index"):
            self._input_index = _input.index
        self._mip_model(**kwargs)
        assert self._output is not None
        return self

    def _mip_model(self, **kwargs):
        """Do the transformation on x."""
        column_transform = self.transformer
        _input = self._input
        transformers = {k.lower(): v for k, v in sklearn_transformers().items()}
        transformed = []
        for name, trans, cols in column_transform.transformers_:
            if trans == "passthrough":
                if isinstance(cols, str) or isinstance(cols[0], str):
                    transformed.append(_input.loc[:, cols])
                else:
                    if hasattr(_input, "iloc"):
                        transformed.append(_input.iloc[:, cols])
                    else:
                        data = _input
                        if data.ndim == 1:
                            # If we have a one-dimensional numpy array reshape it.
                            # By definition pandas dataframe should be 2-dimensional
                            data = data.reshape(1, -1)
                        if isinstance(data, gp.MVar):
                            transformed.append(data[:, cols].tolist())
                        else:
                            transformed.append(data[:, cols])
            elif trans == "drop":
                pass
            else:
                if isinstance(cols, str) or isinstance(cols[0], str):
                    data = _input.loc[:, cols]
                    any_var = any(
                        map(lambda i: isinstance(i, gp.Var), data.to_numpy().ravel())
                    )
                else:
                    if hasattr(_input, "iloc"):
                        data = _input.iloc[:, cols]
                        any_var = any(
                            map(
                                lambda i: isinstance(i, gp.Var), data.to_numpy().ravel()
                            )
                        )
                    else:
                        data = _input
                        if data.ndim == 1:
                            # If we have a one-dimensional numpy array reshape it.
                            # By definition pandas dataframe should be 2-dimensional
                            data = data.reshape(1, -1)
                        data = data[:, cols]
                        any_var = True
                if any_var:
                    try:
                        trans_constr = transformers[name](self.gp_model, trans, data)
                    except KeyError:
                        raise NoModel(name, "No implementation found")
                    transformed.append(trans_constr.output.tolist())
                else:
                    transformed.append(trans.transform(_input.loc[:, cols]))
        self._output = column_transform._hstack(transformed)


def add_column_transformer_constr(gp_model, column_transformer, input_vars, **kwargs):
    """Formulate column_transformer in gurobipy model.

    Parameters
    ----------
    gp_model : :gurobipy:`model`
        The gurobipy model where polynomial features should be inserted.
    column_transformer : :external+sklearn:py:class:`sklearn.compose.ColumnTransformer`
        The column transformer to insert in gp_model.
    input_vars : mvar_array_like
        Decision variables used as input for polynomial features in model.

    Returns
    -------
    sklearn.preprocessing.ColumnTransformerConstr
        Object containing information about what was added to gp_model to insert the
        polynomial_features in it.
    """
    return ColumnTransformerConstr(gp_model, column_transformer, input_vars, **kwargs)
