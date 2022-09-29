# Copyright © 2022 Gurobi Optimization, LLC
"""Internal module to make MIP modeling of activation functions"""

import numpy as np
from gurobipy import GRB


def _name(index, name):
    index = f"{index}".replace(" ", "")
    return f"{name}[{index}]"


def get_mixing(layer, index):
    """Facilitator to compute:
     _input @ layer.coefs + layer.intercept
     for a given index
    Should not be necesary when MVar dimensions are fixed
    """
    k, j = index
    _input = layer.input
    if layer.coefs is None:
        return _input
    input_size = _input.shape[1]
    return sum(_input[k, i] * layer.coefs[i, j] for i in range(input_size)) + layer.intercept[j]


class Identity:
    """Class to apply identity activation on a neural network layer

    Parameters
    ----------
    setbounds: Bool
        Optional flag not to set bounds on the output variables.

    Attributes
    ----------
    setbounds: Bool
        Optional flag not to set bounds on the output variables.
    """

    def __init__(self, setbounds=True):
        self.setbounds = setbounds

    def mip_model(self, layer):
        """MIP model for identity activation on a layer

        Parameters
        ----------
        layer: AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        if self.setbounds:
            output.LB = np.maximum(output.LB, layer.wmin)
            output.UB = np.minimum(output.UB, layer.wmax)

        for index in np.ndindex(output.shape):
            layer.model.addConstr(output[index] == get_mixing(layer, index), name=_name(index, "mix"))

    @staticmethod
    def reset_bounds(layer):
        """Reset the bounds in layer

        Parameters
        ----------
        layer: AbstractNNLayer
            Layer to which activation is applied.
        """
        layer.output.UB = layer.wmax
        layer.output.LB = layer.wmin


class ReLUGC:
    """Class to apply the ReLU activation on a neural network layer

    Parameters
    ----------
    setbounds: Bool
        Optional flag not to set bounds on the output variables.
    bigm: Float
        Optional maximal value for bounds use in the formulation

    Attributes
    ----------
    setbounds: Bool
        Optional flag not to set bounds on the output variables.
    bigm: Float
        Optional maximal value for bounds use in the formulation
    """

    def __init__(self, bigm=None, setbounds=True):
        self.bigm = bigm
        self.setbounds = setbounds

    def mip_model(self, layer):
        """MIP model for ReLU activation on a layer

        Parameters
        ----------
        layer: AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        if hasattr(layer, "coefs"):
            if not hasattr(layer, "mixing"):
                mixing = layer.model.addMVar(output.shape, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="_mix")
                layer.mixing = mixing
            layer.model.update()
            if self.bigm is not None:
                layer.wmax = np.minimum(layer.wmax, self.bigm)
                layer.wmin = np.maximum(layer.wmin, -1 * self.bigm)

            if self.setbounds and layer.wmax is not None:
                output.LB = 0.0
                output.UB = np.maximum(layer.wmax, 0.0)
                layer.mixing.LB = layer.wmin
                layer.mixing.UB = layer.wmax
            for index in np.ndindex(output.shape):
                layer.model.addConstr(layer.mixing[index] == get_mixing(layer, index), name=_name(index, "mix"))
            mixing = layer.mixing
        else:
            mixing = layer._input
        for index in np.ndindex(output.shape):
            layer.model.addGenConstrMax(
                output[index],
                [
                    mixing[index],
                ],
                constant=0.0,
                name=_name(index, "relu"),
            )

    @staticmethod
    def forward(input_values):
        """Return ReLU of input_values"""
        return np.maximum(0.0, input_values)

    @staticmethod
    def forward_fixing(layer, input_values, threshold=-20):
        """Fix binaries according to input_values"""
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
        """Reset the bounds in layer"""
        layer.mixing.UB = layer.wmax
        layer.mixing.LB = layer.wmin


class LogitPWL:
    """Model Logit in a MIP using some PWL formulation"""

    def __init__(self):
        self.zerologit = 1e-1
        self.trouble = 15
        self.nbreak = 15
        self.logitapprox = "3pieces"

    @staticmethod
    def preprocess(layer):
        """Prepare to add logit activation to layer"""

    def _logit_pwl_3pieces(self, xvar, yvar):
        """Do a 3 pieces approximation of logit"""
        zero = self.zerologit
        yval = np.array([zero, 1.0 - zero])
        xval = np.log(yval / (1 - yval))

        if xvar.UB < xval[0]:
            yvar.UB = 0.0
            return (np.array([]), np.array([]))
        if xvar.LB > xval[-1]:
            yvar.LB = 1.0
            return (np.array([]), np.array([]))
        if xvar.LB <= xval[0]:
            xval = np.concatenate(([xvar.LB], xval))
            yval = np.concatenate(([0], yval))
        else:
            xval[0] = xvar.LB
            yval[0] = 1 / (1 + np.exp(-xvar.LB))

        if xvar.UB >= xval[-1]:
            xval = np.concatenate((xval, [xvar.UB]))
            yval = np.concatenate((yval, [1.0]))
        else:
            xval[-1] = xvar.UB
            yval[-1] = 1 / (1 + np.exp(-xvar.UB))
        return (xval, yval)

    def _logit_pwl_approx(self, xvar, yvar):
        """Do a piecewise approximation of logit"""
        shiftlb = False
        shiftub = False
        lower = xvar.LB
        upper = xvar.UB
        careful = True
        if careful:
            if upper < -self.trouble:
                yvar.UB = 0.0
                return (np.array([]), np.array([]))
            if lower > self.trouble:
                yvar.LB = 1.0
                return (np.array([]), np.array([]))

            if lower < -self.trouble:
                shiftlb = True
                lower = -self.trouble
            if upper > self.trouble:
                shiftub = True
                upper = self.trouble
        beg = 1 / (1 + np.exp(-lower))
        end = 1 / (1 + np.exp(-upper))

        yval = np.linspace(beg, end, self.nbreak)
        xval = np.minimum(np.log(yval / (1 - yval)), 1e10)
        xval = np.maximum(xval, -1e10)
        if shiftlb:
            xval = np.concatenate(([xvar.LB], xval))
            yval = np.concatenate(([0], yval))
        if shiftub:
            xval = np.concatenate((xval, [xvar.UB]))
            yval = np.concatenate((yval, [1.0]))

        return (xval, yval)

    def mip_model(self, layer):
        """Add formulation for logit for neuron of layer"""
        model = layer.model
        output = layer.output

        if not layer.zvar:
            zvar = layer.model.addMVar(output.shape, lb=-GRB.INFINITY, name="z")
            layer.zvar = zvar
        else:
            zvar = layer.zvar
        zvar.LB = layer.wmin
        zvar.UB = layer.wmax
        model.update()

        for index in np.ndindex(output.shape):
            mixing = get_mixing(layer, index)
            model.addConstr(layer.zvar[index] == mixing, name=f"_mix[{index}]")
            vact = output[index]
            xvar = layer.zvar[index]
            if self.logitapprox == "PWL":
                xval, yval = self._logit_pwl_approx(xvar, vact)
            else:
                xval, yval = self._logit_pwl_3pieces(xvar, vact)
            if len(xval) > 0:
                layer.model.addGenConstrPWL(xvar, vact, xval, yval, name=f"pwl[{index}]")
