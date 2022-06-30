# Copyright © 2022 Gurobi Optimization, LLC
'''Internal module to make MIP modeling of activation functions'''

# pylint: disable=C0103

import numpy as np
from gurobipy import GRB


def _name(layer, index, name):
    index = f'{index}'.replace(' ','')
    return f'__{layer.name}_{name}[{index}]'


def get_mixing(layer, index):
    '''Facilitator to compute:
        _input @ layer.coefs + layer.intercept
        for a given index
       Should not be necesary when MVar dimensions are fixed
    '''
    k, j = index
    _input = layer._input # pylint: disable=W0212
    input_size = _input.shape[1]
    return sum(_input[k, i] * layer.coefs[i, j]
                     for i in range(input_size)) + layer.intercept[j]

class Identity():
    '''Model identity activation (i.e. does nearly nothing'''
    def __init__(self, setbounds=True):
        self.setbounds = setbounds

    def mip_model(self, layer):
        '''MIP model for identity activation (just apply afine transformation'''
        output = layer._output # pylint: disable=W0212
        if self.setbounds:
            output.LB = np.maximum(output.LB, layer.wmin)
            output.UB = np.minimum(output.UB, layer.wmax)

        for index in np.ndindex(output.shape):
            layer.model.addConstr(output[index] == get_mixing(layer, index),
                                  name=_name(layer, index, 'mix'))

    @staticmethod
    def forward(input_values):
        '''Return input_values'''
        return input_values

    @staticmethod
    def forward_fixing(layer, input_values, threshold=-20):   # pylint: disable=W0613
        '''Fix variables according to input_values noop'''
        return []

    @staticmethod
    def reset_bounds(layer):   # pylint: disable=W0613
        '''Reset the bounds in layer'''


class ReLUGC():
    ''' Model the ReLU function (i.e max(x, 0)) using
        Gurobi max general constraints.'''
    def __init__(self, bigm=None, setbounds=True):
        self.bigm = bigm
        self.setbounds = setbounds

    def mip_model(self, layer):
        ''' Add MIP formulation for ReLU for neuron in layer'''
        output = layer._output # pylint: disable=W0212
        if not hasattr(layer, 'mixing'):
            mixing = layer.model.addMVar(output.shape, lb=-GRB.INFINITY,
                                         vtype=GRB.CONTINUOUS,
                                         name=f'{layer.name}]+_mix')
            layer.mixing = mixing
        layer.model.update()
        if self.bigm is not None:
            layer.wmax = np.minimum(layer.wmax, self.bigm)
            layer.wmin = np.maximum(layer.wmin, -1*self.bigm)
        if self.setbounds and layer.wmax is not None:
            output.LB = 0.0
            output.UB = np.maximum(layer.wmax, 0.0)
            layer.mixing.LB = layer.wmin
            layer.mixing.UB = layer.wmax

        for index in np.ndindex(output.shape):
            mixing = get_mixing(layer, index)
            layer.model.addConstr(layer.mixing[index] == mixing, name=_name(layer, index, 'mix'))
            layer.model.addGenConstrMax(output[index], [layer.mixing[index], ], constant=0.0,
                                        name=_name(layer, index, 'relu'))

    @staticmethod
    def forward(input_values):
        '''Return ReLU of input_values'''
        return np.maximum(0.0, input_values)

    @staticmethod
    def forward_fixing(layer, input_values, threshold=-20):
        '''Fix binaries according to input_values'''
        mixing = layer.mixing
        if threshold < 0:
            threshold = -int(threshold)
            threshold = np.sort(np.abs(input_values))[0, threshold]
        mixing[input_values < 0.0].UB = 0
        mixing[input_values >= 0.0].LB = 0

        closetozero = mixing[np.abs(input_values) <= threshold].tolist()
        return closetozero

    @staticmethod
    def reset_bounds(layer):
        '''Reset the bounds in layer corresponding to modeling ReLU'''
        layer.mixing.UB = layer.wmax
        layer.mixing.LB = layer.wmin


class LogitPWL:
    '''Model Logit in a MIP using some PWL formulation'''
    def __init__(self):
        self.zerologit = 1e-1
        self.trouble = 15
        self.nbreak = 15
        self.logitapprox = '3pieces'

    @staticmethod
    def preprocess(layer):
        '''Prepare to add logit activation to layer'''

    def _logit_pwl_3pieces(self, vx, vy):
        ''' Do a 3 pieces approximation of logit'''
        zero = self.zerologit
        yval = np.array([zero, 1.0 - zero])
        xval = np.log(yval/(1-yval))

        if vx.UB < xval[0]:
            vy.UB = 0.0
            return (np.array([]), np.array([]))
        if vx.LB > xval[-1]:
            vy.LB = 1.0
            return (np.array([]), np.array([]))
        if vx.LB <= xval[0]:
            xval = np.concatenate(([vx.LB], xval))
            yval = np.concatenate(([0], yval))
        else:
            xval[0] = vx.LB
            yval[0] = 1/(1+np.exp(-vx.LB))

        if vx.UB >= xval[-1]:
            xval = np.concatenate((xval, [vx.UB]))
            yval = np.concatenate((yval, [1.0]))
        else:
            xval[-1] = vx.UB
            yval[-1] = 1/(1+np.exp(-vx.UB))
        return (xval, yval)

    def _logit_pwl_approx(self, vx, vy):
        '''Do a piecewise approximation of logit'''
        shiftlb = False
        shiftub = False
        lb = vx.LB
        ub = vx.UB
        careful = True
        if careful:
            if ub < -self.trouble:
                vy.UB = 0.0
                return (np.array([]), np.array([]))
            if lb > self.trouble:
                vy.LB = 1.0
                return (np.array([]), np.array([]))

            if lb < -self.trouble:
                shiftlb = True
                lb = -self.trouble
            if ub > self.trouble:
                shiftub = True
                ub = self.trouble
        beg = 1/(1+np.exp(-lb))
        end = 1/(1+np.exp(-ub))

        yval = np.linspace(beg, end, self.nbreak)
        xval = np.minimum(np.log(yval/(1-yval)), 1e10)
        xval = np.maximum(xval, -1e10)
        if shiftlb:
            xval = np.concatenate(([vx.LB], xval))
            yval = np.concatenate(([0], yval))
        if shiftub:
            xval = np.concatenate((xval, [vx.UB]))
            yval = np.concatenate((yval, [1.0]))

        return (xval, yval)

    def mip_model(self, layer):
        '''Add formulation for logit for neuron of layer'''
        model = layer.model
        output = layer._output # pylint: disable=W0212
        _input = layer._input # pylint: disable=W0212

        if not layer.zvar:
            z = layer.model.addMVar(output.shape, lb=-GRB.INFINITY,
                                    name=f'__z[{layer.name}]')
            layer.zvar = z
        else:
            z = layer.zvar
        z.LB = layer.wmin
        z.UB = layer.wmax
        model.update()

        for index in np.ndindex(output.shape):
            mixing = get_mixing(layer, index)
            model.addConstr(layer.zvar[index] == mixing, name=f'{layer.name}_mix[{index}]')
            vact = output[index]
            vx = layer.zvar[index]
            if self.logitapprox == 'PWL':
                xval, yval = self._logit_pwl_approx(vx, vact)
            else:
                xval, yval = self._logit_pwl_3pieces(vx, vact)
            if len(xval) > 0:
                layer.model.addGenConstrPWL(vx, vact, xval, yval,
                                            name=f'{layer.name}_pwl[{index}]')
