# Copyright © 2022 Gurobi Optimization, LLC
''' Base module to instert constraint based on neural network
or regrssion constructs in Gurobi'''

# pylint: disable=C0103

import numpy as np
import gurobipy as gp

from .activations2grb import LogitPWL
from .activations2grb import ReLUGC
from .activations2grb import Identity
from .utils import validate_gpvars, transpose


class NNLayer:
    '''Class to build one layer of a neural network'''

    def __init__(self, model, activation_vars, input_vars, layer_coefs,
                 layer_intercept, activation_function, name):
        self.model = model
        self.actvar = activation_vars
        self.invar = input_vars
        self.coefs = layer_coefs
        self.intercept = layer_intercept
        self.activation = activation_function
        self.wmin = None
        self.wmax = None
        self.zvar = None
        self.constrs = []
        self.name = name

    def getmixing(self, index):
        '''Mix layer input'''
        (k, j) = index
        input_size = self.invar.shape[1]

        mixing = sum(self.invar[k, i] * self.coefs[i, j]
                     for i in range(input_size)) + self.intercept[j]
        return mixing

    def getname(self, index, name):
        '''Get a fancy name for a neuron in layer'''
        return f'{name}[{self.name}][{index[0]},{index[1]}]'

    def _wminmax(self):
        '''Compute min/max for w variable'''
        if (self.invar.UB >= - gp.GRB.INFINITY).any():
            return (None, None)
        if (self.invar.LB <= gp.GRB.INFINITY).any():
            return (None, None)
        wpos = np.maximum(self.coefs, 0.0)
        wneg = np.minimum(self.coefs, 0.0)
        wmin = self.invar.LB @ wpos + self.invar.UB @ wneg + self.intercept
        wmax = self.invar.UB @ wpos + self.invar.LB @ wneg + self.intercept
        wmax = np.maximum(wmin, wmax)

        return (wmin, wmax)

    def add(self, activation=None):
        ''' Add the layer to model'''
        model = self.model
        model.update()
        activation_vars = self.actvar
        input_vars = self.invar
        layer_coefs = self.coefs
        if activation is None:
            activation = self.activation
        (n, _) = input_vars.shape
        layer_size = layer_coefs.shape[1]

        # Add activation variables if we don't have them
        if activation_vars is None:
            activation_vars = model.addMVar((input_vars.shape[0], layer_coefs.shape[1]),
                                            lb=-gp.GRB.INFINITY,
                                            name=f'__act[{self.name}]')
            self.actvar = activation_vars
        assert layer_size == activation_vars.shape[1]
        assert n == activation_vars.shape[0]

        # Compute bounds on weighted sums by propagation
        if activation.setbounds:
            wmin, wmax = self._wminmax()
            if wmax is not None and self.wmax is not None:
                wmax = np.minimum(wmax, self.wmax)
            if wmin is not None and self.wmin is not None:
                wmin = np.maximum(wmin, self.wmin)
            self.wmin = wmin
            self.wmax = wmax

        # Apply bounds to activation variables (and other preprocessing)
        activation.preprocess(self)

        # Now build model neuron by neuron and example by example
        for j in range(layer_size):
            for k in range(n):
                activation.conv(self, (k, j))
        self.model.update()

    def reset_bounds(self):
        '''Reset bounds on layer'''
        activation_function = self.activation
        model = self.model
        model.update()

        activation_function.reset_bounds(self)

        self.model.update()

    def redolayer(self, activation=None):
        ''' Rebuild the layer (possibly using a different model for activation)'''
        self.model.remove(self.constrs)
        self.add(activation)


class BaseNNRegression2Grb:
    ''' Base class for inserting a regressor based on neural-network/tensor into Gurobi'''

    def __init__(self, regressor, model, name='', clean_regressor=False):
        self.regressor = regressor
        self.model = model
        self.name = name
        self.clean = clean_regressor
        self.actdict = {'relu': ReLUGC(), 'identity': Identity(), 'logit': LogitPWL()}
        self._input = None
        self._output = None
        self._layers = []
        self._cuts = []

    def __iter__(self):
        return self._layers.__iter__()

    @staticmethod
    def validate(input_vars, output_vars):
        input_vars = validate_gpvars(input_vars)
        output_vars = validate_gpvars(output_vars)
        if output_vars.shape[0] != input_vars.shape[0] and output_vars.shape[1] != input_vars.shape[0]:
            raise BaseException("Non-conforming dimension between input variable and output variable: {} != {}".
                                format(output_vars.shape[0], input_vars.shape[0]))
        elif output_vars.shape[1] == input_vars.shape[0]:
            output_vars = transpose(output_vars)

        return (input_vars, output_vars)

    def addlayer(self, input_vars, layer_coefs,
                 layer_intercept, activation,
                 activation_vars=None, name=None):
        '''add a layer to model'''
        if name is None:
            name = f'{len(self._layers)}'
        name = f'{self.name}[{name}]'

        layer = NNLayer(self.model, activation_vars, input_vars, layer_coefs,
                        layer_intercept, activation, name)
        if self.clean:
            mask = np.abs(layer.coefs) < 1e-8
            layer.coefs[mask] = 0.0
        layer.add()
        self._layers.append(layer)
        return layer

    def rebuild_formulation(self, activation=None):
        '''Rebuild the MIP formulation for regression model'''
        for layer in self:
            layer.redolayer(activation)
        self.model.update()
