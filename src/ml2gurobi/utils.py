# Copyright © 2022 Gurobi Optimization, LLC
import types  # NOQA

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


model_stats = {'NumConstrs': 'getConstrs',
              'NumQConstrs': 'getQConstrs',
              'NumVars': 'getVars',
              'NumGenConstrs': 'getGenConstrs'}


class Submodel(object):
    def __init__(self, model):
        self.model = model
        self.torec_ = {name: func
                       for name, func in model_stats.items()}
        self.added_ = {name: [] for name in model_stats.keys()}
        for stat, func in model_stats.items():
            exec(f'''def {func}(self): return self.added_['{stat}']''')
            exec(f'self.{func} = types.MethodType({func}, self)')
            exec(f'self.{stat} = 0')


    def get_stats_(self):
        m = self.model
        m.update()
        rval = {}
        for s in self.torec_.keys():
            rval[s] = m.getAttr(s)
        return rval

    @staticmethod
    def validate(input_vars, output_vars):
        input_vars = validate_gpvars(input_vars)
        output_vars = validate_gpvars(output_vars)
        if output_vars.shape[0] != input_vars.shape[0] and output_vars.shape[1] != input_vars.shape[0]:
            raise BaseException("Non-conforming dimension between input variable and output variable: {} != {}".
                             format(output_vars.shape[0], input_vars.shape[0]))
        elif input_vars.shape[0] != output_vars.shape[0] and output_vars.shape[1] == input_vars.shape[0]:
            output_vars = transpose(output_vars)

        return (input_vars, output_vars)

    def update(self, begin, end):
        for s in self.torec_.keys():
            added = end[s] - begin[s]
            if added == 0:
                continue
            self.added_[s] += (eval('self.model.' + self.torec_[s] + '()')[begin[s]: end[s]])
            exec(f'self.{s} += added')

    def mip_model(self, X, y):
        pass

    def predict(self, X, y):
        begin = self.get_stats_()
        X, y = self.validate(X, y)
        self.mip_model(X, y)
        end = self.get_stats_()
        self.update(begin, end)

    def remove(self):
        for s, v in self.added_.items():
            self.model.remove(v)
            self.added_[s] = []
        self.model.update()
