# Copyright © 2023-2026 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Internal module to make MIP modeling of activation functions."""

import numpy as np
from gurobipy import GRB

from ...exceptions import NoModel

try:
    from gurobipy import nlfunc

    _HAS_NLEXPR = True
except ImportError:
    _HAS_NLEXPR = False

from ..softmax import hardmax, softmax


class Identity:
    """Class to apply identity activation on a neural network layer.

    Parameters
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.

    Attributes
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    """

    def __init__(self):
        pass

    def mip_model(self, layer):
        """MIP model for identity activation on a layer.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        layer.gp_model.addConstr(output == layer.input @ layer.coefs + layer.intercept)
        layer.linear_predictor = output


class ReLU:
    """Class to apply the ReLU activation on a neural network layer.

    Parameters
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    bigm : Float
        Optional maximal value for bounds use in the formulation

    Attributes
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    bigm : Float
        Optional maximal value for bounds use in the formulation
    """

    def __init__(self):
        pass

    def mip_model(self, layer):
        """MIP model for ReLU activation on a layer.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        if hasattr(layer, "coefs"):
            if not hasattr(layer, "linear_predictor"):
                linear_predictor = layer.gp_model.addMVar(
                    output.shape,
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=layer._name_var("mix"),
                )
                layer.linear_predictor = linear_predictor
            layer.gp_model.update()

            layer.gp_model.addConstr(
                layer.linear_predictor == layer.input @ layer.coefs + layer.intercept
            )
        else:
            linear_predictor = layer._input

        for index in np.ndindex(output.shape):
            layer.gp_model.addGenConstrMax(
                output[index],
                [
                    linear_predictor[index],
                ],
                constant=0.0,
                name=layer._indexed_name(index, "relu"),
            )


class Logistic:
    """Class to apply the logistic activation on a neural network layer.

    Parameters
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    bigm : Float
        Optional maximal value for bounds use in the formulation

    Attributes
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    bigm : Float
        Optional maximal value for bounds use in the formulation
    """

    def __init__(self):
        if not _HAS_NLEXPR:
            raise NoModel(self, "Can't use logistic activation without Gurobi ≥ 12.0")

    def mip_model(self, layer):
        """MIP model for logistic activation on a layer.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        if hasattr(layer, "coefs"):
            layer.linear_predictor = layer.input @ layer.coefs + layer.intercept
            linear_predictor = layer.linear_predictor
        else:
            linear_predictor = layer._input
        layer.gp_model.addConstr(layer.output == nlfunc.logistic(linear_predictor))


class SoftMax:
    """Class to apply the SoftMax activation on a neural network layer.

    Parameters
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    bigm : Float
        Optional maximal value for bounds use in the formulation

    Attributes
    ----------
    setbounds : Bool
        Optional flag not to set bounds on the output variables.
    bigm : Float
        Optional maximal value for bounds use in the formulation
    """

    def __init__(self):
        pass

    def mip_model(self, layer):
        """MIP model for SoftMax activation on a layer.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        if hasattr(layer, "coefs"):
            layer.linear_predictor = layer.input @ layer.coefs + layer.intercept
            linear_predictor = layer.linear_predictor
        else:
            linear_predictor = layer._input

        if hasattr(layer, "predict_function"):
            predict_function = layer.predict_function
            if predict_function == "predict_proba":
                softmax(layer, linear_predictor)
            elif predict_function == "predict":
                hardmax(layer, linear_predictor)
            elif predict_function == "decision_function":
                output == linear_predictor
        else:
            softmax(layer, linear_predictor)
