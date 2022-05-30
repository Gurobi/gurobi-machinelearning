# Copyright © 2022 Gurobi Optimization, LLC

'''Utilities for ml2gurobi'''

# pylint: disable=C0103

import gurobipy as gp


def validate_gpvars(gpvars):
    ''' Put variables into appropriate form (matrix of variable)'''
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
        return gp.MVar([gpvars, ], shape=(1, 1))
    raise BaseException("Could not validate variables")


def transpose(gpvars):
    '''Transpose a matrix of variables

    Should I really do this?
    '''
    assert isinstance(gpvars, gp.MVar)
    assert gpvars.ndim == 2
    return gp.MVar(gpvars.tolist()[0], (gpvars.shape[1], gpvars.shape[0]))


model_stats = {'NumConstrs': 'Constrs',
               'NumQConstrs': 'QConstrs',
               'NumVars': 'Vars',
               'NumSOS': 'SOSs',
               'NumGenConstrs': 'GenConstrs'}


def addtosubmodel(function):
    ''' Wrapper function to add to submodel '''
    def wrapper(self, *args, **kwargs):
        begin = self.get_stats_()
        function(self, *args, **kwargs)
        end = self.get_stats_()
        self.update(begin, end)
    return wrapper


class Submodel:
    ''' Class to define a submodel'''
    def __init__(self, model, name=''):
        self.model = model
        self.name = name
        self.torec_ = {'NumConstrs': model.getConstrs,
                       'NumQConstrs': model.getQConstrs,
                       'NumVars': model.getVars,
                       'NumSOS':  model.getSOSs,
                       'NumGenConstrs': model.getGenConstrs}
        for stat, func in model_stats.items():
            self.__dict__[func] = []
            self.__dict__[stat] = 0

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
        '''Update submodel after changes

        (i.e. record what happened between begin and end).
        '''
        for s, func in self.torec_.items():
            added = end[s] - begin[s]
            if added == 0:
                continue
            self.__dict__[s] += added
            added = func()[begin[s]: end[s]]
            self.__dict__[model_stats[s]] += added

    def mip_model(self, X, y):
        ''' Mip model for predict
        (implemented in child classes)'''

    @addtosubmodel
    def predict(self, X, y):
        '''Predict y from X using regression/classifier

        X and y are Gurobi Matrices or list of variables of conforming
        dimensions'''
        X, y = self.validate(X, y)
        self.mip_model(X, y)

    def remove(self, what=None):
        '''Remove everything added by this object from Gurobi model'''
        if what is None:
            for s, v in model_stats.items():
                self.model.remove(self.__dict__[v])
                self.__dict__[v] = []
                self.__dict__[s] = 0
        else:
            for s in what:
                key = 'Num'+s
                self.model.remove(self.__dict__[key])
                self.__dict__[s] = []
                self.__dict__[key] = 0
        self.model.update()
