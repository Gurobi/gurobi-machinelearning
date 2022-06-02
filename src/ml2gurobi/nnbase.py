# Copyright © 2022 Gurobi Optimization, LLC
''' Base module to instert constraint based on neural network
or regrssion constructs in Gurobi'''

# pylint: disable=C0103

import gurobipy as gp
import numpy as np

from .activations import Identity, LogitPWL, ReLUGC
from .utils import MLSubModel, addtosubmodel


class NNLayer(MLSubModel):
    '''Class to build one layer of a neural network'''

    def __init__(self, model, output_vars, input_vars, layer_coefs,
                 layer_intercept, activation_function, name):
        super().__init__(model, input_vars, output_vars, name)
        self.coefs = layer_coefs
        self.intercept = layer_intercept
        self.activation = activation_function
        self.wmin = None
        self.wmax = None
        self.zvar = None

    def getname(self, index, name):
        '''Get a fancy name for a neuron in layer'''
        return f'{name}[{self.name}][{index[0]},{index[1]}]'

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

    @addtosubmodel
    def add(self, activation=None):
        ''' Add the layer to model'''
        model = self.model
        model.update()
        output = self._output
        _input = self._input  # pylint: disable=W0212
        layer_coefs = self.coefs
        if activation is None:
            activation = self.activation
        n, _ = _input.shape
        _, layer_size = layer_coefs.shape

        # Add activation variables if we don't have them
        if output is None:
            output = model.addMVar((n, layer_size),
                                            lb=-gp.GRB.INFINITY,
                                            name=f'__act[{self.name}]')
            self._output = output

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
        self.add(activation)


class BaseNNRegression2Gurobi(MLSubModel):
    ''' Base class for inserting a regressor based on neural-network/tensor into Gurobi'''

    def __init__(self, model, regressor, input_vars, output_vars, name='', clean_regressor=False):
        super().__init__(model, input_vars, output_vars, name)
        self.regressor = regressor
        self.clean = clean_regressor
        self.actdict = {'relu': ReLUGC(), 'identity': Identity(), 'logit': LogitPWL()}
        self._layers = []

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
            if not isinstance(layer.activation, Identity):
                layer.redolayer(activation)
        self.model.update()
