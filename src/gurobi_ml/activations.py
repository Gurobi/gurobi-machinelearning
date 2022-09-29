# Copyright © 2022 Gurobi Optimization, LLC
"""Internal module to make MIP modeling of activation functions"""

import numpy as np
from gurobipy import GRB


def _name(index, name):
    index = f"{index}".replace(" ", "")
    return f"{name}[{index}]"


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

    def __init__(self):
        pass

    def mip_model(self, layer):
        """MIP model for identity activation on a layer

        Parameters
        ----------
        layer: AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        layer.model.addConstr(output == layer.input @ layer.coefs + layer.intercept)

    @staticmethod
    def reset_bounds(layer):
        """Reset the bounds in layer

        Parameters
        ----------
        layer: AbstractNNLayer
            Layer to which activation is applied.
        """
        output.LB = -GRB.Infinity
        output.UB = GRB.Infinity


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

    def __init__(self):
        pass

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

            layer.model.addConstr(layer.mixing == layer.input @ layer.coefs + layer.intercept)
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
        layer.mixing.UB = GRB.Infinity
        layer.mixing.LB = -GRB.Infinity
        layer.output.UB = GRB.Infinity
        layer.output.LB = -GRB.Infinity


class LogisticGC:
    """Model Logit in a MIP using some PWL formulation"""

    def __init__(self, gc_attributes=None):
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

            mixing = layer.mixing
            layer.model.addConstr(mixing == layer.input @ layer.coefs + layer.intercept)
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
