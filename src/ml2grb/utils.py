# Copyright © 2022 Gurobi Optimization, LLC
import gurobipy as gp


def validate_gpvars(gpvars):
    if isinstance(gpvars, gp.MVar):
        if gpvars.ndim == 1:
            return gp.MVar(gpvars.tolist(), shape=(1, gpvars.shape[0]))
        if gpvars.ndim == 2:
            return gpvars
        else:
            raise BaseException("Variables should be an MVar of dimension 1 or 2")
    if isinstance(gpvars, dict):
        gpvars = gpvars.values()
    if isinstance(gpvars, list):
        return gp.MVar(gpvars, shape=(1, len(gpvars)))
    if isinstance(gpvars, gp.Var):
        rval = gp.MVar(gpvars, shape=(1, 1))
        rval._vararr = rval._vararr.reshape((1, 1))  # Bug in MVar? an MVar of a single var doesn't follow shape
        return rval


def transpose(gpvars):
    assert isinstance(gpvars, gp.MVar)
    assert gpvars.ndim == 2
    return gp.MVar(gpvars.tolist()[0], (gpvars.shape[1], gpvars.shape[0]))
