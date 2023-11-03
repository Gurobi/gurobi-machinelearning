# Copyright © 2023 Gurobi Optimization, LLC
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
:gurobipy:`model`.
"""

import numpy as np
import torch
from torch import nn

from ..exceptions import NoModel, NoSolution
from ..modeling.neuralnet import BaseNNConstr


def add_sequential_constr(
    gp_model, sequential_model, input_vars, output_vars=None, **kwargs
):
    """Formulate sequential_model into gp_model.

    The formulation predicts the values of output_vars using input_vars according to sequential_model.
    See our :ref:`Users Guide <Neural Networks>` for details on the mip formulation used.

    Parameters
    ----------
    gp_model : :gurobipy:`model`
        The gurobipy model where the sequential model should be inserted.
    sequential_model : :external+torch:py:class:`torch.nn.Sequential`
        The sequential model to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for logistic regression in model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for logistic regression in model.

    Returns
    -------
    SequentialConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Raises
    ------
    NoModel
        If the translation for some of the Pytorch model structure
        (layer or activation) is not implemented.

    Warnings
    --------
    Only :external+torch:py:class:`torch.nn.Linear` layers and
    :external+torch:py:class:`torch.nn.ReLU` layers are supported.

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
        for step in predictor:
            if isinstance(step, nn.ReLU):
                pass
            elif isinstance(step, nn.Linear):
                pass
            else:
                raise NoModel(predictor, f"Unsupported layer {type(step).__name__}")
        super().__init__(gp_model, predictor, input_vars, output_vars)

    def _mip_model(self, **kwargs):
        network = self.predictor
        _input = self._input
        output = None
        num_layers = len(network)

        for i, step in enumerate(network):
            if i == num_layers - 1:
                output = self._output
            if isinstance(step, nn.ReLU):
                layer = self._add_activation_layer(
                    _input, self.act_dict["relu"], output, name=f"relu_{i}", **kwargs
                )
                _input = layer.output
            elif isinstance(step, nn.Linear):
                for name, param in step.named_parameters():
                    if name == "weight":
                        layer_weight = param.detach().numpy().T
                    elif name == "bias":
                        layer_bias = param.detach().numpy()
                layer = self._add_dense_layer(
                    _input,
                    layer_weight,
                    layer_bias,
                    self.act_dict["identity"],
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
                print(f"{t_out} != {self.output_values}")
            return r_val
        raise NoSolution()
