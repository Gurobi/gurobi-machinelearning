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


class ReLU:
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
