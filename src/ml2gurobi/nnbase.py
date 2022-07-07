# Copyright © 2022 Gurobi Optimization, LLC
''' Base module to instert constraint based on neural network
or regrssion constructs in Gurobi'''

# pylint: disable=C0103

import gurobipy as gp
import numpy as np

from .activations import Identity, LogitPWL, ReLUGC
from .utils import AbstractPredictor


class NNLayer(AbstractPredictor):
    '''Class to build one layer of a neural network'''
    def __init__(self, model, output_vars, input_vars, layer_coefs,
                 layer_intercept, activation_function, name,**kwargs):
        self.coefs = layer_coefs
        self.intercept = layer_intercept
        self.activation = activation_function
        self.wmin = None
        self.wmax = None
        self.zvar = None
        super().__init__(model, input_vars, output_vars, name, **kwargs)

    def getname(self, index):
        '''Get a fancy name for a neuron in layer'''
        return f'{self.name}][{index[0]},{index[1]}'

    def _wminmax(self):
        '''Compute min/max for w variable'''
        if (self._input.UB >= gp.GRB.INFINITY).any():
            return (-gp.GRB.INFINITY*np.ones(self._output.shape),
                    gp.GRB.INFINITY*np.ones(self._output.shape))
        if (self._input.LB <= - gp.GRB.INFINITY).any():
            return (-gp.GRB.INFINITY*np.ones(self._output.shape),
                    gp.GRB.INFINITY*np.ones(self._output.shape))
        wpos = np.maximum(self.coefs, 0.0)
        wneg = np.minimum(self.coefs, 0.0)
        wmin = self._input.LB @ wpos + self._input.UB @ wneg + self.intercept
        wmax = self._input.UB @ wpos + self._input.LB @ wneg + self.intercept
        wmax = np.maximum(wmin, wmax)

        return (wmin, wmax)

    def mip_model(self, activation=None):
        ''' Add the layer to model'''
        model = self.model
        model.update()
        _input = self._input  # pylint: disable=W0212
        layer_coefs = self.coefs
        if activation is None:
            activation = self.activation
        n, _ = _input.shape
        _, layer_size = layer_coefs.shape

        # Compute bounds on weighted sums by propagation
        wmin, wmax = self._wminmax()

        # Take best bound from what we have stored and what we propagated
        if wmax is not None and self.wmax is not None:
            wmax = np.minimum(wmax, self.wmax)
        if wmin is not None and self.wmin is not None:
            wmin = np.maximum(wmin, self.wmin)
        self.wmin = wmin
        self.wmax = wmax

        # Do the mip model for the activation in the layer
        activation.mip_model(self)
        self.model.update()

    def reset_bounds(self):
        '''Reset bounds on layer'''
        activation_function = self.activation
        self.model.update()
        activation_function.reset_bounds(self)
        self.model.update()

    def redolayer(self, activation=None):
        ''' Rebuild the layer (possibly using a different model for activation)'''
        self.remove(['Constrs', 'QConstrs', 'GenConstrs'])
        self._add(activation)


class BaseNNPredictor(AbstractPredictor):
    ''' Base class for inserting a regressor based on neural-network/tensor into Gurobi'''

    def __init__(self, model, regressor, input_vars, output_vars, name='', clean_regressor=False, **kwargs):
        self.regressor = regressor
        self.clean = clean_regressor
        self.actdict = {'relu': ReLUGC(), 'identity': Identity(), 'logit': LogitPWL()}
        self._layers = []
        super().__init__(model, input_vars, output_vars, name, **kwargs)

    def __iter__(self):
        return self._layers.__iter__()

    def addlayer(self, input_vars, layer_coefs,
                 layer_intercept, activation,
                 activation_vars=None, name=None):
        '''Add a layer to model'''
        if name is None:
            name = f'{len(self._layers)}'
        if self.name != '':
            name = f'{self.name}[{name}]'

        # Add activation variables if we don't have them
        if activation_vars is None:
            activation_vars = self.model.addMVar((input_vars.shape[0], layer_coefs.shape[1]),
                                                 lb=-gp.GRB.INFINITY,
                                                 name=f'__act[{self.name}]')
        if self.clean:
            mask = np.abs(layer.coefs) < 1e-8
            layer.coefs[mask] = 0.0
        layer = NNLayer(self.model, activation_vars, input_vars, layer_coefs,
                        layer_intercept, activation, name)
        self._layers.append(layer)
        return layer

    def rebuild_formulation(self, activation=None):
        '''Rebuild the MIP formulation for regression model'''
        for layer in self:
            if not isinstance(layer.activation, Identity):
                layer.redolayer(activation)
        self.model.update()

    def reset_bounds(self):
        ''' Reset bounds of variables in mip model'''
        for layer in self:
            layer.reset_bounds()
        self.model.update()
