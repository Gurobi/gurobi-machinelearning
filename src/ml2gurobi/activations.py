# Copyright © 2022 Gurobi Optimization, LLC
'''Internal module to make MIP modeling of activation functions'''

# pylint: disable=C0103

import numpy as np
from gurobipy import GRB


class Identity():
    '''Model identity activation (i.e. does nearly nothing'''
    def __init__(self, setbounds=True):
        self.setbounds = setbounds

    def mip_model(self, layer):
        if self.setbounds:
            layer.actvar.LB = np.maximum(layer.actvar.LB, layer.wmin)
            layer.actvar.UB = np.minimum(layer.actvar.UB, layer.wmax)

        input_size = layer.invar.shape[1]
        for index in np.ndindex(layer.actvar.shape):
            (k, j) = index
            mixing = sum(layer.invar[k, i] * layer.coefs[i, j]
                      for i in range(input_size)) + layer.intercept[j]
            layer.model.addConstr(layer.actvar[index] == mixing, name=layer.name+f'_mix[{index}]')


class ReLUGC():
    ''' Model the ReLU function (i.e max(x, 0)) using
        Gurobi max general constraints.'''
    def __init__(self, bigm=None, setbounds=True):
        self.bigm = bigm
        self.setbounds = setbounds

    def mip_model(self, layer):
        ''' Add MIP formulation for ReLU for neuron in layer'''
        if not hasattr(layer, 'mixing'):
            mixing = layer.model.addMVar(layer.actvar.shape, lb=-GRB.INFINITY,
                                         vtype=GRB.CONTINUOUS,
                                         name=f'__mix[{layer.name}]')
            layer.mixing = mixing
        layer.model.update()
        input_size = layer.invar.shape[1]
        if self.bigm is not None:
            layer.wmax = np.minimum(layer.wmax, self.bigm)
            layer.wmin = np.maximum(layer.wmin, -1*self.bigm)
        if self.setbounds and layer.wmax is not None:
            layer.actvar.LB = 0.0
            layer.actvar.UB = np.maximum(layer.wmax, 0.0)
            layer.mixing.LB = layer.wmin
            layer.mixing.UB = layer.wmax

        for index in np.ndindex(layer.actvar.shape):
            (k, j) = index
            mixing = sum(layer.invar[k, i] * layer.coefs[i, j]
                      for i in range(input_size)) + layer.intercept[j]
            layer.model.addConstr(layer.mixing[index] == mixing, name=layer.name+f'_mix[{index}]')
            layer.model.addGenConstrMax(layer.actvar[index], [layer.mixing[index], ], constant=0.0, name=layer.name+'_relu')


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
        if not layer.zvar:
            z = layer.model.addMVar(layer.actvar.shape, lb=-GRB.INFINITY,
                                    name=f'__z[{layer.name}]')
            layer.zvar = z
        else:
            z = layer.zvar
        z.LB = layer.wmin
        z.UB = layer.wmax
        model.update()
        mixing = layer.invar @ layer.coefs + layer.intercept
        model.addConstr(layer.zvar == mixing, name=layer.name+'_mix')

        for index in np.ndindex(layer.actvar.shape):
            vact = layer.actvar[index]
            vx = layer.zvar[index]
            if self.logitapprox == 'PWL':
                xval, yval = self._logit_pwl_approx(vx, vact)
            else:
                xval, yval = self._logit_pwl_3pieces(vx, vact)
            if len(xval) > 0:
                layer.model.addGenConstrPWL(vx, vact, xval, yval,
                                        name=constrname+'_pwl')
