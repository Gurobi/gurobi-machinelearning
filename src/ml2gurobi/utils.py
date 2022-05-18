# Copyright © 2022 Gurobi Optimization, LLC

# pylint: disable=C0103

import types  # NOQA

import gurobipy as gp


def validate_gpvars(gpvars):
    if isinstance(gpvars, gp.MVar):
        if gpvars.ndim == 1:
            return gp.MVar(gpvars.tolist(), shape=(1, gpvars.shape[0]))
        if gpvars.ndim == 2:
            return gpvars
        raise BaseException("Variables should be an MVar of dimension 1 or 2")
    if isinstance(gpvars, dict):
        gpvars = gpvars.values()
    if isinstance(gpvars, list):
        return gp.MVar(gpvars, shape=(1, len(gpvars)))
    if isinstance(gpvars, gp.Var):
        rval = gp.MVar(gpvars, shape=(1, 1))
        # Bug in MVar? an MVar of a single var doesn't follow shape
        rval._vararr = rval._vararr.reshape((1, 1))
        return rval
    raise BaseException("Could not validate variables")


def transpose(gpvars):
    assert isinstance(gpvars, gp.MVar)
    assert gpvars.ndim == 2
    return gp.MVar(gpvars.tolist()[0], (gpvars.shape[1], gpvars.shape[0]))


model_stats = {'NumConstrs': 'getConstrs',
               'NumQConstrs': 'getQConstrs',
               'NumVars': 'getVars',
               'NumSOS': 'getSOSs',
               'NumGenConstrs': 'getGenConstrs'}

name_attrs = {'NumConstrs': 'ConstrName',
              'NumQConstrs': 'QCName',
              'NumVars': 'VarName',
              'NumSOS': 'SOSName',
              'NumGenConstrs': 'GenConstrName'}

def addtosubmodel(function):
    def wrapper(self, *args, **kwargs):
        begin = self.get_stats_()
        function(self, *args, **kwargs)
        end = self.get_stats_()
        self.update(begin, end)
    return wrapper


class Submodel:
    def __init__(self, model, name=''):
        self.model = model
        self.name = name
        self.torec_ = model_stats.copy()
        self.added_ = {name: [] for name in model_stats}
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
        ''' Validate input and output variables (check shapes, reshape if needed.'''
        input_vars = validate_gpvars(input_vars)
        output_vars = validate_gpvars(output_vars)
        if (output_vars.shape[0] != input_vars.shape[0] and
            output_vars.shape[1] != input_vars.shape[0]):
            raise BaseException("Non-conforming dimension between " +
                                "input variable and output variable: {} != {}".
                                format(output_vars.shape[0], input_vars.shape[0]))
        if (input_vars.shape[0] != output_vars.shape[0] and
            output_vars.shape[1] == input_vars.shape[0]):
            output_vars = transpose(output_vars)

        return (input_vars, output_vars)

    def update(self, begin, end):
        for s in self.torec_.keys():
            added = end[s] - begin[s]
            if added == 0:
                continue
            exec(f'self.{s} += added')
            added = (eval('self.model.' + self.torec_[s] + '()')[begin[s]: end[s]])
            # if self.name != '':
            #     for o in added:
            #         oname = o.getAttr(name_attrs[s])
            #         o.setAttr(name_attrs[s], self.name+oname)
            self.added_[s] += added

    def mip_model(self, X, y):
        pass

    @addtosubmodel
    def predict(self, X, y):
        X, y = self.validate(X, y)
        self.mip_model(X, y)

    def remove(self, what=None):
        if what is None:
            for s, v in self.added_.items():
                self.model.remove(v)
                self.added_[s] = []
        else:
            for s in what:
                key = 'Num'+s
                self.model.remove(self.added_[key])
                self.added_[key] = []
        self.model.update()
