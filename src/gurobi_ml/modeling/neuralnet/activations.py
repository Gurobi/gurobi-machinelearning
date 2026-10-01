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


def _linear_bounds(layer):
    """Compute bounds on ``layer.input @ layer.coefs + layer.intercept``.

    The bounds are obtained by interval arithmetic from the bounds of the
    input variables of the layer. They can be infinite.
    """
    input_lb = layer.input.getAttr(GRB.Attr.LB)
    input_ub = layer.input.getAttr(GRB.Attr.UB)
    coefs_pos = np.maximum(layer.coefs, 0.0)
    coefs_neg = np.minimum(layer.coefs, 0.0)

    # Do the computation on finite bounds and record separately where an
    # infinite bound contributes (avoids 0 * inf).
    lb_inf = input_lb <= -GRB.INFINITY
    ub_inf = input_ub >= GRB.INFINITY
    input_lb = np.where(lb_inf, 0.0, input_lb)
    input_ub = np.where(ub_inf, 0.0, input_ub)

    lb = input_lb @ coefs_pos + input_ub @ coefs_neg + layer.intercept
    ub = input_ub @ coefs_pos + input_lb @ coefs_neg + layer.intercept
    lb[(lb_inf @ (coefs_pos > 0) + ub_inf @ (coefs_neg < 0)) > 0] = -GRB.INFINITY
    ub[(ub_inf @ (coefs_pos > 0) + lb_inf @ (coefs_neg < 0)) > 0] = GRB.INFINITY
    return lb, ub


def _bound_output(layer, lb, ub):
    """Impose bounds lb and ub on the output variables of layer.

    Bounds are set directly on variables that were created for the layer.
    On variables provided by the user we add constraints instead so that the
    bounds of their variables are not modified.
    """
    output = layer.output
    if getattr(layer, "_output_created", False):
        output.LB = np.maximum(output.LB, lb)
        output.UB = np.minimum(output.UB, ub)
        return
    finite = lb > -GRB.INFINITY
    if finite.any():
        layer.gp_model.addConstr(output[finite] >= lb[finite])
    finite = ub < GRB.INFINITY
    if finite.any():
        layer.gp_model.addConstr(output[finite] <= ub[finite])


class Identity:
    """Class to apply identity activation on a neural network layer.

    Parameters
    ----------
    setbounds : Bool
        Optional flag to set bounds on the output variables. The bounds are
        derived from the bounds of the input variables of the layer.

    Attributes
    ----------
    setbounds : Bool
        Optional flag to set bounds on the output variables.
    """

    def __init__(self, setbounds=False):
        self.setbounds = setbounds

    def mip_model(self, layer):
        """MIP model for identity activation on a layer.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which activation is applied.
        """
        output = layer.output
        if self.setbounds:
            _bound_output(layer, *_linear_bounds(layer))
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


class ReLUBigM:
    """Class to apply the ReLU activation on a neural network layer using
    big-M constraints instead of general constraints.

    For each neuron with pre-activation value :math:`a \\in [L, U]` and
    output :math:`y`, a binary variable :math:`z` is introduced and

    .. math::

        y \\geq a, \\quad y \\geq 0, \\quad y \\leq a - L (1 - z), \\quad y \\leq U z.

    If bigm is not given, the bounds :math:`L` and :math:`U` are computed from
    the bounds of the input variables of the layer. Neurons that are always
    active or always inactive don't need a binary variable.

    If bigm is given, :math:`L = -M` and :math:`U = M` are used for all neurons
    and each neuron gets a binary variable. This gives a weaker formulation
    that is valid only if :math:`|a| \\leq M`.

    Parameters
    ----------
    bigm : float, optional
        Value :math:`M` used for all neurons. If it is not given and some
        bounds are infinite, a ValueError is raised.

    Attributes
    ----------
    bigm : float or None
        Value :math:`M` used for all neurons.
    """

    def __init__(self, bigm=None):
        self.bigm = bigm

    def mip_model(self, layer):
        """MIP model for ReLU activation on a layer.

        Parameters
        ----------
        layer : AbstractNNLayer
            Layer to which activation is applied.
        """
        model = layer.gp_model
        output = layer.output
        model.update()
        if hasattr(layer, "coefs"):
            mixing = model.addMVar(
                output.shape,
                lb=-GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name=layer._name_var("mix"),
            )
            layer.mixing = mixing
            model.addConstr(mixing == layer.input @ layer.coefs + layer.intercept)
        else:
            mixing = layer.input

        if self.bigm is not None:
            # Same big-M for all neurons
            lb = np.full(output.shape, -float(self.bigm))
            ub = np.full(output.shape, float(self.bigm))
            _bound_output(
                layer, np.zeros(output.shape), np.full(output.shape, GRB.INFINITY)
            )
        else:
            if hasattr(layer, "coefs"):
                lb, ub = _linear_bounds(layer)
            else:
                lb = mixing.getAttr(GRB.Attr.LB)
                ub = mixing.getAttr(GRB.Attr.UB)
            if (lb <= -GRB.INFINITY).any() or (ub >= GRB.INFINITY).any():
                raise ValueError(
                    "Big-M formulation of ReLU requires finite bounds on the "
                    "input of the neural network or a value for bigm"
                )
            _bound_output(layer, np.maximum(lb, 0.0), np.maximum(ub, 0.0))

        active = lb >= 0.0
        undecided = (lb < 0.0) & (ub > 0.0)
        # Inactive neurons have output fixed to 0 by the bounds above
        if active.any():
            model.addConstr(output[active] == mixing[active])
        if undecided.any():
            lb, ub = lb[undecided], ub[undecided]
            y, a = output[undecided], mixing[undecided]
            z = model.addMVar(
                y.shape, vtype=GRB.BINARY, name=layer._name_var("relu_active")
            )
            layer.zvar = z
            model.addConstr(y >= a)
            model.addConstr(y <= a - lb * (1 - z))
            model.addConstr(y <= ub * z)
