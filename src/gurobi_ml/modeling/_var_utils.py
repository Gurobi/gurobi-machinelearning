# Copyright © 2023-2025 Gurobi Optimization, LLC
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


def _ndarray_to_mvar_general(model, data):
    """Convert an arbitrary-dimensional numpy array mixing gp.Var and constants
    into a gp.MVar with the same shape.

    For entries that are constants, create fixed variables (lb=ub=value).
    For entries that are gp.Var, reuse them.
    The construction is done on a flattened 1D list and reshaped back.
    """
    arr = np.asarray(data, dtype=object)
    flat = arr.ravel().tolist()
    is_var = [isinstance(x, gp.Var) for x in flat]
    if all(is_var):
        mv = gp.MVar.fromlist(flat)
        return mv.reshape(arr.shape)
    # Build list of constants (as floats)
    const_vals = []
    const_pos = []
    for idx, x in enumerate(flat):
        if not is_var[idx]:
            try:
                const_vals.append(float(x))
            except Exception as e:
                raise ValueError(
                    f"Entry at position {idx} can't be converted to float: {x}"
                ) from e
            const_pos.append(idx)
    # Create fixed vars for constants
    if const_vals:
        const_m = model.addMVar(len(const_vals))
        const_m.LB = np.array(const_vals, dtype=float)
        const_m.UB = np.array(const_vals, dtype=float)
        model.update()
    # Rebuild the flat gp.Var list in original order
    rebuilt = []
    ci = 0
    for idx, x in enumerate(flat):
        if is_var[idx]:
            rebuilt.append(x)
        else:
            rebuilt.append(const_m[ci])
            ci += 1
    mv = gp.MVar.fromlist(rebuilt)
    return mv.reshape(arr.shape)


def validate_output_vars(gp_vars, accepted_dim=(1, 2)):
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
    # Pass-through if already an MVar of accepted shape
    if isinstance(gp_vars, gp.MVar):
        # If 1D outputs are acceptable, keep 1D so caller can orient later.
        if gp_vars.ndim == 1 and 1 in accepted_dim:
            return gp_vars
        if gp_vars.ndim in accepted_dim:
            return gp_vars
        # If 1D not accepted but 2D is, promote to row vector
        if gp_vars.ndim == 1 and 2 in accepted_dim:
            return gp_vars.reshape(1, -1)
        # Try to add a leading batch dimension if that makes it valid
        if (gp_vars.ndim + 1) in accepted_dim:
            if gp_vars.ndim == 1:
                return gp_vars.reshape(1, -1)
            if gp_vars.ndim == 3:
                return gp_vars.reshape((1,) + gp_vars.shape)
        raise ParameterError(
            "Variables should be an MVar of dimension {}".format(
                " or ".join([f"{d}" for d in accepted_dim])
            )
        )

    # Pandas supported only for 1D/2D output (tabular)
    if HAS_PANDAS and isinstance(gp_vars, (pd.DataFrame, pd.Series)):
        if not any(d in accepted_dim for d in (1, 2)):
            raise ParameterError(
                "Pandas outputs only supported for 1D/2D outputs; got accepted_dim="
                + f"{accepted_dim}"
            )
        return validate_output_vars(gp_vars.to_numpy(), accepted_dim=accepted_dim)

    # Numpy arrays of vars supported for any dimension
    if isinstance(gp_vars, np.ndarray):
        if gp_vars.size == 0:
            raise ParameterError("Empty output variable array is not supported")
        if all(isinstance(v, gp.Var) for v in gp_vars.ravel()):
            mv = gp.MVar.fromlist(list(gp_vars.ravel()))
            mv = mv.reshape(gp_vars.shape)
        else:
            raise TypeError("Output arrays must contain only gp.Var entries")
        # Adjust shape if needed: keep 1D if accepted, else promote to 2D if allowed
        if mv.ndim == 1 and 1 in accepted_dim:
            return mv
        if mv.ndim in accepted_dim:
            return mv
        if mv.ndim == 1 and 2 in accepted_dim:
            return mv.reshape(1, -1)
        if (mv.ndim + 1) in accepted_dim:
            if mv.ndim == 1:
                return mv.reshape(1, -1)
            if mv.ndim == 3:
                return mv.reshape((1,) + mv.shape)
        raise ParameterError(
            "Output variables have dimension {} but expected {}".format(
                mv.ndim, ", ".join(map(str, accepted_dim))
            )
        )

    # Lists/dicts/Var: treat as 1D and adjust
    if isinstance(gp_vars, dict):
        gp_vars = list(gp_vars.values())
    if isinstance(gp_vars, list):
        mv = gp.MVar.fromlist(gp_vars)
        return validate_output_vars(mv, accepted_dim=accepted_dim)
    if isinstance(gp_vars, gp.Var):
        return validate_output_vars(
            gp.MVar.fromlist([gp_vars]), accepted_dim=accepted_dim
        )
    raise ParameterError("Could not validate output variables")


def validate_input_vars(model, gp_vars, accepted_dim=(1, 2)):
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
    # If already an MVar, adjust shape if needed and return
    if isinstance(gp_vars, gp.MVar):
        mv = gp_vars
        # Prefer adding a batch dimension for 1D when 2D is accepted
        if mv.ndim == 1 and 2 in accepted_dim:
            return (mv.reshape(1, -1), None, None)
        if mv.ndim in accepted_dim:
            return (mv, None, None)
        # Try to add a leading batch dimension if that makes it valid
        if (mv.ndim + 1) in accepted_dim:
            if mv.ndim == 1:
                return (mv.reshape(1, -1), None, None)
            if mv.ndim == 3:
                return (mv.reshape((1,) + mv.shape), None, None)
        raise ParameterError(
            "Variables should be an MVar of dimension {} and is dimension {}".format(
                " or ".join([f"{d}" for d in accepted_dim]), mv.ndim
            )
        )

    # Pandas supported for 1D/2D predictors only
    if HAS_PANDAS and isinstance(gp_vars, (pd.DataFrame, pd.Series)):
        if not any(d in accepted_dim for d in (1, 2)):
            raise ParameterError(
                "Pandas inputs only supported for 1D/2D predictors; got accepted_dim="
                + f"{accepted_dim}"
            )
        columns = gp_vars.columns
        index = gp_vars.index
        mv = _dataframe_to_mvar(model, gp_vars)
        # Ensure 2D (batch, features)
        if mv.ndim == 1:
            mv = mv.reshape(1, -1)
        if mv.ndim not in (1, 2):
            raise ParameterError("DataFrame inputs must be 1D or 2D")
        return (mv, columns, index)

    # Numpy arrays: support any dimensionality
    if isinstance(gp_vars, np.ndarray):
        # For 1D/2D, reuse the optimized conversion to fix constants by columns
        if gp_vars.ndim in (1, 2):
            mv = _array_to_mvar(model, gp_vars)
        else:
            mv = _ndarray_to_mvar_general(model, gp_vars)
        # Adjust shape for accepted dims (prefer 2D for 1D when allowed)
        if mv.ndim == 1 and 2 in accepted_dim:
            return (mv.reshape(1, -1), None, None)
        if mv.ndim in accepted_dim:
            return (mv, None, None)
        if (mv.ndim + 1) in accepted_dim:
            if mv.ndim == 1:
                return (mv.reshape(1, -1), None, None)
            if mv.ndim == 3:
                return (mv.reshape((1,) + mv.shape), None, None)
        raise ParameterError(
            "Input variables have dimension {} but expected {}".format(
                mv.ndim, ", ".join(map(str, accepted_dim))
            )
        )

    # dict/list/Var fallbacks -> 1D vector
    if isinstance(gp_vars, dict):
        gp_vars = list(gp_vars.values())
    if isinstance(gp_vars, list):
        mv = gp.MVar.fromlist(gp_vars)
        return validate_input_vars(model, mv, accepted_dim=accepted_dim)
    if isinstance(gp_vars, gp.Var):
        mv = gp.MVar.fromlist([gp_vars])
        return validate_input_vars(model, mv, accepted_dim=accepted_dim)
    raise ParameterError("Could not validate input variables")
