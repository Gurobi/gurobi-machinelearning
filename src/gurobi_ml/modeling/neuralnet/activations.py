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


class _NLActivation:
    """Base class for smooth nonlinear activations using Gurobi nlfunc (Gurobi 12+).

    Subclasses implement :meth:`_nl_expr` to define the per-neuron nonlinear
    expression and :attr:`_constraint_name` to provide a label for the
    generated constraints.  All boilerplate for creating the pre-activation
    mixing variables and iterating over neurons lives here.
    """

    #: Label used when naming the nonlinear constraints (override in subclass).
    _constraint_name: str = "nl"

    def __init__(self):
        if nlfunc is None:
            raise RuntimeError(
                f"{type(self).__name__} requires Gurobi 12.0+ with nonlinear "
                "function support. Please upgrade Gurobi or use a MIP-compatible "
                "activation (e.g. ReLU)."
            )

    def _nl_expr(self, x):
        """Return the nonlinear nlfunc expression for scalar pre-activation ``x``.

        Parameters
        ----------
        x : gurobipy Var
            A single Gurobi variable representing the pre-activation value.

        Returns
        -------
        gurobipy nonlinear expression
        """
        raise NotImplementedError

    def mip_model(self, layer):
        """Add nonlinear activation constraints to *layer*.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which the activation is applied.
        """
        output = layer.output
        if hasattr(layer, "coefs"):
            if not hasattr(layer, "mixing"):
                layer.mixing = layer.gp_model.addMVar(
                    output.shape,
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=layer._name_var("mix"),
                )
            layer.gp_model.update()
            layer.gp_model.addConstr(
                layer.mixing == layer.input @ layer.coefs + layer.intercept
            )
            mixing = layer.mixing
        else:
            mixing = layer._input

        for index in np.ndindex(output.shape):
            layer.gp_model.addConstr(
                output[index] == self._nl_expr(mixing[index]),
                name=layer._indexed_name(index, self._constraint_name),
            )



class Sigmoid(_NLActivation):
    """Sigmoid activation via ``nlfunc.logistic``: f(x) = 1 / (1 + exp(-x)).

    Requires Gurobi 12.0+ with nonlinear function support.
    """

    _constraint_name = "sigmoid"

    def _nl_expr(self, x):
        return nlfunc.logistic(x)


class Tanh(_NLActivation):
    """Tanh activation via ``nlfunc.tanh``: f(x) = tanh(x).

    Requires Gurobi 12.0+ with nonlinear function support.
    """

    _constraint_name = "tanh"

    def _nl_expr(self, x):
        return nlfunc.tanh(x)


class SoftPlus(_NLActivation):
    """Softplus activation: f(x) = (1/β) · log(1 + exp(β·x)).

    A smooth approximation of ReLU compatible with Gurobi's nonlinear barrier
    solver (Gurobi 12+).

    Parameters
    ----------
    beta : float, optional
        Sharpness parameter (default 1.0). Higher values make the curve closer
        to ReLU.
    """

    _constraint_name = "soft_relu"

    def __init__(self, beta=1.0):
        super().__init__()
        if beta <= 0.0:
            raise ValueError("beta must be strictly positive")
        self.beta = beta

    def _nl_expr(self, x):
        return (1.0 / self.beta) * nlfunc.log(1.0 + nlfunc.exp(self.beta * x))
