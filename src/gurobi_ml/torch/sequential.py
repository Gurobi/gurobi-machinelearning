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

"""Module for formulating :external+torch:py:class:`torch.nn.Sequential` model in a
:external+gurobi:py:class:`Model`.
"""

import numpy as np
import torch
import warnings
from torch import nn
import gurobipy as gp

from ..exceptions import ModelConfigurationError, NoSolutionError
from ..modeling.neuralnet import BaseNNConstr, SoftPlus


# Check Gurobi version
GUROBI_VERSION = gp.gurobi.version()
HAS_NLFUNC = GUROBI_VERSION >= (12, 0, 0)

# Map nn.Module activation types to their activation-registry name.
# nn.Softplus is handled separately because it carries parameters (beta, threshold).
_NN_ACTIVATION_MAP = {
    nn.ReLU: "relu",
    nn.Sigmoid: "sigmoid",
}
if HAS_NLFUNC:
    _NN_ACTIVATION_MAP[nn.Tanh] = "tanh"


def add_sequential_constr(
    gp_model, sequential_model, input_vars, output_vars=None, **kwargs
):
    """Formulate sequential_model into gp_model.

    The formulation predicts the values of output_vars using input_vars according to sequential_model.
    See our :ref:`Users Guide <Neural Networks>` for details on the mip formulation used.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the sequential model should be inserted.
    sequential_model : :external+torch:py:class:`torch.nn.Sequential`
        The sequential model to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for pytorch model in gp_model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for pytorch model in gp_model.

    Returns
    -------
    SequentialConstr
        Object containing information about what was added to gp_model to insert the
        predictor in it

    Raises
    ------
    ModelConfigurationError
        If the translation for some of the Pytorch model structure
        (layer or activation) is not implemented.

    Warnings
    --------
    Only :external+torch:py:class:`torch.nn.Linear` layers,
    :external+torch:py:class:`torch.nn.ReLU`,
    :external+torch:py:class:`torch.nn.Sigmoid`,
    :external+torch:py:class:`torch.nn.Tanh`, and
    :external+torch:py:class:`torch.nn.Softplus` layers are supported.

    Notes
    -----
    |VariablesDimensionsWarn|
    """
    return SequentialConstr(
        gp_model, sequential_model, input_vars, output_vars, **kwargs
    )


class SequentialConstr(BaseNNConstr):
    """Transform a pytorch Sequential Neural Network to Gurobi constraint with
    input and output as matrices of variables.
    |ClassShort|.
    """

    def __init__(self, gp_model, predictor, input_vars, output_vars=None, **kwargs):
        _supported = (*_NN_ACTIVATION_MAP, nn.Softplus, nn.Linear)
        for step in predictor:
            if not isinstance(step, _supported):
                raise ModelConfigurationError(
                    predictor, f"Unsupported layer {type(step).__name__}"
                )
        super().__init__(gp_model, predictor, input_vars, output_vars, **kwargs)

    def _mip_model(self, **kwargs):
        network = self.predictor
        _input = self._input
        output = None
        num_layers = len(network)

        for i, step in enumerate(network):
            if i == num_layers - 1:
                output = self._output

            act_name = _NN_ACTIVATION_MAP.get(type(step))
            if act_name is not None:
                layer = self._add_activation_layer(
                    _input,
                    self._get_activation(act_name),
                    output,
                    name=f"{act_name}_{i}",
                    **kwargs,
                )
                _input = layer.output
            elif isinstance(step, nn.Softplus):
                # Softplus carries parameters; validate threshold before constructing.
                if step.threshold != 20:
                    raise ModelConfigurationError(
                        self.predictor,
                        f"PyTorch Softplus with non-default threshold ({step.threshold}) is not supported. "
                        f"Only threshold=20 (default) is supported.",
                    )
                layer = self._add_activation_layer(
                    _input,
                    SoftPlus(beta=step.beta),
                    output,
                    name=f"softplus_{i}",
                    **kwargs,
                )
                _input = layer.output
            elif isinstance(step, nn.Linear):
                layer_weight = None
                layer_bias = None
                for name, param in step.named_parameters():
                    if name == "weight":
                        layer_weight = param.detach().numpy().T
                    elif name == "bias":
                        layer_bias = param.detach().numpy()
                if layer_weight is None:
                    raise NotImplementedError("No weights specified for newwork layer.")
                if layer_bias is None:
                    layer_bias = 0.0
                layer = self._add_dense_layer(
                    _input,
                    layer_weight,
                    layer_bias,
                    self._get_activation("identity"),
                    output,
                    name=f"linear_{i}",
                    **kwargs,
                )
                _input = layer.output
        if self._output is None:
            self._output = layer.output

    def get_error(self, eps=None):
        if self._has_solution:
            t_in = torch.from_numpy(self.input_values).float()
            t_out = self.predictor.forward(t_in)
            r_val = np.abs(t_out.detach().numpy() - self.output_values)
            if eps is not None and np.max(r_val) > eps:
                warnings.warn(f"get_error: {t_out} != {self.output_values}")
            return r_val
        raise NoSolutionError()
