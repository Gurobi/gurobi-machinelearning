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


def addtosubmodel(function):
    ''' Wrapper function to add to submodel '''
    def wrapper(self, *args, **kwargs):
        begin = self.get_stats_()
        rval = function(self, *args, **kwargs)
        end = self.get_stats_()
        self.update(begin, end)
        return rval
    return wrapper


class AbstractPredictor:
    ''' Class to define a submodel'''
    def __init__(self, model, input_vars, output_vars, name='', delayed_add=False, **kwargs):
        self.model = model
        self.name = name
        self.torec_ = {'Constrs': model.getConstrs,
                       'QConstrs': model.getQConstrs,
                       'Vars': model.getVars,
                       'SOS':  model.getSOSs,
                       'GenConstrs': model.getGenConstrs}

        self.list_ = {}
        self.num_ = {}
        for stat in self.torec_:
            self.list_[stat] = []
            self.num_[stat] = 0
        if input_vars is not None:
            self._set_input(input_vars)
        else:
            self._input = None
        if output_vars is not None:
            self._set_output(output_vars)
        else:
            self._output = None
        self._delayed_add = (input_vars is None or
                             output_vars is None or
                             delayed_add)
        if not self._delayed_add:
            self._add()

    def get_stats_(self):
        ''' Get model's statistics'''
        m = self.model
        m.update()
        rval = {}
        for s in self.torec_:
            rval[s] = m.getAttr(f'Num{s}')
        return rval

    def get_num(self, stat):
        return self.num_[stat]

    def get_list(self, stat):
        return self.list_[stat]

    def _set_input(self, input_vars):
        self._input = validate_gpvars(input_vars)

    def _set_output(self, output_vars):
        self._output = validate_gpvars(output_vars)

    @staticmethod
    def validate(input_vars, output_vars):
        ''' Validate input and output variables (check shapes, reshape if needed.'''
        if input_vars is None:
            raise BaseException('No input variables')
        if output_vars is None:
            raise BaseException('No output variables')

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
            self.num_[s] += added
            added = func()[begin[s]: end[s]]
            self.list_[s] += added

    @addtosubmodel
    def _add(self):
        '''Predict output from input using regression/classifier'''
        self._input, self._output = self.validate(self._input, self._output)
        self.mip_model()
        return self

    def remove(self, what=None):
        '''Remove everything added by this object from Gurobi model'''
        if what is None:
            for s in self.torec_:
                self.model.remove(self.list_[s])
                self.list_[s] = []
                self.num_[s] = 0
        else:
            for key in what:
                self.model.remove(self.list_[key])
                self.num_[key] = 0
                self.list_[key] = []
        self.model.update()
