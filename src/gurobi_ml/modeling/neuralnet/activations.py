# Copyright © 2022 Gurobi Optimization, LLC
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

"""Internal module to make MIP modeling of activation functions"""

import numpy as np
from gurobipy import GRB


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
        layer.gp_model.addConstr(output == layer.input @ layer.coefs + layer.intercept)


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
                mixing = layer.gp_model.addMVar(
                    output.shape, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="_mix"
                )
                layer.mixing = mixing
            layer.gp_model.update()

            layer.gp_model.addConstr(layer.mixing == layer.input @ layer.coefs + layer.intercept)
        else:
            mixing = layer._input
        for index in np.ndindex(output.shape):
            layer.gp_model.addGenConstrMax(
                output[index],
                [
                    mixing[index],
                ],
                constant=0.0,
                name=layer._indexed_name(index, "relu"),
            )
