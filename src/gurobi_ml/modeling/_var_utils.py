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

"""Utility functions for dealing with input and output variables of predictor constr objects."""

import gurobipy as gp
import numpy as np

from ..exceptions import ParameterError

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _default_name(predictor):
    """Make a default name for predictor constraint.

    Parameters
    ----------
    predictor:
        Class of the predictor
    """
    return type(predictor).__name__.lower()


def _get_sol_values(values, columns=None, index=None):
    """Get solution values.

    This is complicated because of the column_transformer.
    In most cases we can just do values.X but if we have a column transformer with
    some constants that can't be translated to Gurobi variables we need to fill in missing values
    """
    if HAS_PANDAS:
        if isinstance(values, pd.DataFrame):
            rval = pd.DataFrame(
                data=_get_sol_values(values.to_numpy()),
                index=values.index,
                columns=values.columns,
            )
            for col in rval.columns:
                try:
                    rval[col] = rval[col].astype(np.float64)
                except ValueError:
                    pass
            return rval.convert_dtypes()
    if isinstance(values, np.ndarray):
        return np.array(
            [v.X if isinstance(v, gp.Var) else v for v in values.ravel()]
        ).reshape(values.shape)
    X = values.X
    if columns is not None and HAS_PANDAS:
        if isinstance(index, (pd.Index, pd.MultiIndex)):
            X = pd.DataFrame(data=X, columns=columns, index=index)
        else:
            X = pd.Series(data=X, columns=columns, name=index)
            raise NotImplementedError("Input variables as pd.Series is not implemented")
    return X


def _dataframe_to_mvar(model, df):
    """Convert a DataFrame to an mvar.

    This just calls the array_to_mvar function
    """
    if isinstance(df, pd.DataFrame):
        data = df.to_numpy()
        columns = df.columns
        index = df.index
    elif isinstance(df, pd.Series):
        raise NotImplementedError("Input variables as pd.Series is not implemented")
    return _array_to_mvar(model, data, columns, index)


def _array_to_mvar(model, data, columns=None, index=None):
    """Function to create an MVar from an ndarray.

    The array data may contain columns of Gurobi variables and constant values.
    In this function we create an array of same shape as data where constant
    values are replace by fixed variables. This is then converted to an MVar.

    If columns and index are passed, used them to give appropriate names to
    created variables.

    Parameters
    ----------
    model : gp.Model
        The gurobipy model
    data : ndarray
        A numpy array whose columns are either columns of gp.Var or data that
        can be converted to a float
    columns : optional
        Name of data columns
    index:
        Name of data rows
    """

    # If data only contains gp.Var's we can directly convert it to an MVar
    if all(map(lambda i: isinstance(i, gp.Var), data.ravel())):
        rval = gp.MVar.fromlist(data.tolist())
        return rval

    # data doesn't contain only contain gp.Var,s.
    # Iterate through the columns, keep columns of variables as is.
    # For other columns check if they can be converted to numbers and collect
    # their indices
    rval = np.zeros(data.shape, dtype=object)
    const_indices = []
    for i, a in enumerate(data.T):
        if all(map(lambda i: isinstance(i, gp.Var), a)):
            rval[:, i] = a
            continue
        try:
            rval[:, i] = a.astype(np.float64)
        except TypeError:
            raise ValueError(
                f"Column {i} of input variables can't be converted to gurobipy variables or floats"
            )
        const_indices.append(i)

    # If columns and index are not passed try to infer something to name the variables
    if columns is None or index is None:
        if data.ndim == 2:
            main_shape = data.shape[1]
            minor_shape = data.shape[0]
        else:
            main_shape = data.shape[0]
            minor_shape = 1
        if columns is None:
            columns = [f"feat{i}" for i in range(main_shape)]
        if index is None:
            index = list(range(minor_shape))

    # Create missing variables, fix them and put them into the array
    mvar = model.addMVar((data.shape[0], len(const_indices)))
    for i, j in enumerate(const_indices):
        mvar[:, i].LB = rval[:, j]
        mvar[:, i].UB = rval[:, j]
        mvar[:, i].VarName = [f"{columns[j]}[{k}]" for k in index]
        rval[:, j] = mvar[:, i].tolist()
    model.update()
    rval = gp.MVar.fromlist(rval)
    return rval


def validate_output_vars(gp_vars):
    """Put variables into appropriate form (matrix of variable) for the output of a predictor constraint.

    Parameters
    ----------
    gpvars:
        Decision variables used.

    Returns
    -------
    mvar_array_like
        Decision variables with correctly adjusted shape.
    """
    if HAS_PANDAS:
        if isinstance(gp_vars, (pd.DataFrame, pd.Series)):
            return validate_output_vars(gp_vars.to_numpy())
    if isinstance(gp_vars, np.ndarray):
        if any(map(lambda i: not isinstance(i, gp.Var), gp_vars.ravel())):
            raise TypeError("Dataframe can't be converted to an MVar")
        rval = gp.MVar.fromlist(gp_vars.tolist())
        return rval
    if isinstance(gp_vars, gp.MVar):
        if gp_vars.ndim in (1, 2):
            return gp_vars
        raise ParameterError("Variables should be an MVar of dimension 1 or 2")
    if isinstance(gp_vars, dict):
        gp_vars = list(gp_vars.values())
    if isinstance(gp_vars, list):
        mvar = gp.MVar.fromlist(gp_vars)
        return validate_output_vars(mvar)
    if isinstance(gp_vars, gp.Var):
        return gp.MVar.fromlist([gp_vars]).reshape(1, 1)
    raise ParameterError("Could not validate variables")


def validate_input_vars(model, gp_vars):
    """Put variables into appropriate form (matrix of variable) for the input of a predictor constraint.

    Parameters
    ----------
    gpvars:
        Decision variables used.

    Returns
    -------
    mvar_array_like
        Decision variables with correctly adjusted shape.
    """
    if HAS_PANDAS:
        if isinstance(gp_vars, (pd.DataFrame, pd.Series)):
            columns = gp_vars.columns
            index = gp_vars.index
            gp_vars = _dataframe_to_mvar(model, gp_vars)
            return (gp_vars, columns, index)
    if isinstance(gp_vars, np.ndarray):
        return (_array_to_mvar(model, gp_vars), None, None)
    if isinstance(gp_vars, gp.MVar):
        if gp_vars.ndim == 1:
            return (gp_vars.reshape(1, -1), None, None)
        if gp_vars.ndim == 2:
            return (gp_vars, None, None)
        raise ParameterError("Variables should be an MVar of dimension 1 or 2")
    if isinstance(gp_vars, dict):
        gp_vars = list(gp_vars.values())
    if isinstance(gp_vars, list):
        mvar = gp.MVar.fromlist(gp_vars)
        return validate_input_vars(model, mvar)
    if isinstance(gp_vars, gp.Var):
        return (gp.MVar.fromlist([gp_vars]).reshape(1, 1), None, None)
    raise ParameterError("Could not validate variables")
