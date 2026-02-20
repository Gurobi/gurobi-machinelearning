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

try:
    from gurobipy import GRB, nlfunc
except ImportError:
    # Fallback for Gurobi versions that do not provide nlfunc (pre-12.0)
    from gurobipy import GRB  # type: ignore[misc]

    nlfunc = None  # type: ignore[assignment]


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
            if not hasattr(layer, "mixing"):
                mixing = layer.gp_model.addMVar(
                    output.shape,
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=layer._name_var("mix"),
                )
                layer.mixing = mixing
            layer.gp_model.update()

            layer.gp_model.addConstr(
                layer.mixing == layer.input @ layer.coefs + layer.intercept
            )
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


class SqrtReLU:
    """Class to apply a ReLU activation using a nonlinear sqrt-based formulation.

    Uses the formulation: f(x) = (x + sqrt(x^2)) / 2, which is
    mathematically equivalent to ReLU since sqrt(x^2) = |x|.

    This representation can be convenient for modeling ReLU with
    Gurobi's non-linear barrier solver (Gurobi 12+), but note that
    it remains non-differentiable at x = 0 and is not a smooth
    approximation of ReLU.
    """

    def __init__(self):
        if nlfunc is None:
            raise RuntimeError(
                "SqrtReLU requires Gurobi 12.0+ with nonlinear function support. "
                "Please upgrade Gurobi or use the standard ReLU formulation."
            )

    def mip_model(self, layer):
        """Sqrt-based ReLU model for activation on a layer using nonlinear expressions.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        if hasattr(layer, "coefs"):
            if not hasattr(layer, "mixing"):
                mixing = layer.gp_model.addMVar(
                    output.shape,
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=layer._name_var("mix"),
                )
                layer.mixing = mixing
            layer.gp_model.update()

            layer.gp_model.addConstr(
                layer.mixing == layer.input @ layer.coefs + layer.intercept
            )
        else:
            mixing = layer._input

        # Use nonlinear expression: output = (x + sqrt(x^2)) / 2
        for index in np.ndindex(output.shape):
            layer.gp_model.addConstr(
                output[index]
                == 0.5 * (mixing[index] + nlfunc.sqrt(mixing[index] * mixing[index])),
                name=layer._indexed_name(index, "smooth_relu"),
            )


class SoftReLU:
    """Class to apply soft ReLU (softplus) activation on a neural network layer.

    Uses the formulation: f(x) = (1/beta) * log(1 + exp(beta * x))
    This is a smooth approximation of ReLU suitable for Gurobi's
    non-linear barrier solver (Gurobi 12+).

    Parameters
    ----------
    beta : float, optional
        Smoothness parameter. Default is 1.0. Higher values make it closer to ReLU.
    """

    def __init__(self, beta=1.0):
        if nlfunc is None:
            raise RuntimeError(
                "SoftReLU requires Gurobi 12.0+ with nonlinear function support. "
                "Please upgrade Gurobi or use the standard ReLU formulation."
            )
        if beta <= 0.0:
            raise ValueError("beta must be strictly positive")
        self.beta = beta

    def mip_model(self, layer):
        """Soft ReLU model for activation on a layer using nonlinear expressions.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        if hasattr(layer, "coefs"):
            if not hasattr(layer, "mixing"):
                mixing = layer.gp_model.addMVar(
                    output.shape,
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=layer._name_var("mix"),
                )
                layer.mixing = mixing
            layer.gp_model.update()

            layer.gp_model.addConstr(
                layer.mixing == layer.input @ layer.coefs + layer.intercept
            )
        else:
            mixing = layer._input

        # Use nonlinear expression: output = (1/beta) * log(1 + exp(beta * x))
        for index in np.ndindex(output.shape):
            layer.gp_model.addConstr(
                output[index]
                == (1.0 / self.beta)
                * nlfunc.log(1.0 + nlfunc.exp(self.beta * mixing[index])),
                name=layer._indexed_name(index, "soft_relu"),
            )
