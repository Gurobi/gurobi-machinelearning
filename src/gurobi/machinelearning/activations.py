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
    """Model identity activation (i.e. does nearly nothing"""

    def __init__(self, setbounds=True):
        self.setbounds = setbounds

    def mip_model(self, layer):
        """MIP model for identity activation (just apply afine transformation"""
        output = layer.output
        if self.setbounds:
            output.LB = np.maximum(output.LB, layer.wmin)
            output.UB = np.minimum(output.UB, layer.wmax)

        for index in np.ndindex(output.shape):
            layer.model.addConstr(output[index] == get_mixing(layer, index), name=_name(index, "mix"))

    @staticmethod
    def forward(input_values):
        """Return input_values"""
        return input_values

    @staticmethod
    def forward_fixing(layer, input_values, threshold=-20):  # pylint: disable=W0613
        """Fix variables according to input_values noop"""
        return []

    @staticmethod
    def reset_bounds(layer):  # pylint: disable=W0613
        """Reset the bounds in layer"""


class ReLUGC:
    """Model the ReLU function (i.e max(x, 0)) using
    Gurobi max general constraints."""

    def __init__(self, bigm=None, setbounds=True):
        self.bigm = bigm
        self.setbounds = setbounds

    def mip_model(self, layer):
        """Add MIP formulation for ReLU for neuron in layer"""
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
        """Reset the bounds in layer corresponding to modeling ReLU"""
        layer.mixing.UB = layer.wmax
        layer.mixing.LB = layer.wmin


class LogisticGC:
    """Model Logit in a MIP using some PWL formulation"""

    def __init__(self, bigm=None, setbounds=True, gc_attributes=None):
        self.bigm = bigm
        self.setbounds = setbounds
        if gc_attributes is None:
            self.attributes = {"FuncPieces": -1, "FuncPieceLength": 0.01, "FuncPieceError": 0.1, "FuncPieceRatio": -1.0}
        else:
            self.attributes = gc_attributes

    def mip_model(self, layer):
        """Add formulation for logit for neuron of layer"""
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
                output.UB = 1.0
                layer.mixing.LB = layer.wmin
                layer.mixing.UB = layer.wmax
            for index in np.ndindex(output.shape):
                layer.model.addConstr(layer.mixing[index] == get_mixing(layer, index), name=_name(index, "mix"))
            mixing = layer.mixing
        else:
            mixing = layer._input
        for index in np.ndindex(output.shape):
            gc = layer.model.addGenConstrLogistic(
                mixing[index],
                output[index],
                name=_name(index, "logistic"),
            )
        numgc = layer.model.NumGenConstrs
        layer.model.update()
        for gc in layer.model.getGenConstrs()[numgc:]:
            for attr, val in self.attributes.items():
                gc.setAttr(attr, val)
