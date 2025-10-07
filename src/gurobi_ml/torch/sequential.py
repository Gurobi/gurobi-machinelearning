# Copyright © 2023-2025 Gurobi Optimization, LLC
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
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the sequential model should be inserted.
    sequential_model : :external+torch:py:class:`torch.nn.Sequential`
        The sequential model to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for model represented by pytorch object.
    output_vars : mvar_array_like, optional
        Decision variables used as output for model represented by pytorch object.

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
    Supported layers:
    :external+torch:py:class:`torch.nn.Linear`,
    :external+torch:py:class:`torch.nn.ReLU`,
    :external+torch:py:class:`torch.nn.Conv2d`,
    :external+torch:py:class:`torch.nn.MaxPool2d`,
    :external+torch:py:class:`torch.nn.Flatten`, and
    :external+torch:py:class:`torch.nn.Dropout` (treated as identity).

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
            elif isinstance(step, nn.Conv2d):
                # Only support padding equivalent to 'valid'
                pad = step.padding
                if (
                    isinstance(pad, str)
                    or (isinstance(pad, tuple) and any(p != 0 for p in pad))
                    or (isinstance(pad, int) and pad != 0)
                ):
                    raise NoModel(
                        predictor,
                        "Only Conv2d with padding=0 ('valid') is supported",
                    )
            elif isinstance(step, nn.MaxPool2d):
                # Only support padding equivalent to 'valid'
                pad = step.padding
                if (
                    isinstance(pad, str)
                    or (isinstance(pad, tuple) and any(p != 0 for p in pad))
                    or (isinstance(pad, int) and pad != 0)
                ):
                    raise NoModel(
                        predictor,
                        "Only MaxPool2d with padding=0 ('valid') is supported",
                    )
            elif isinstance(step, nn.Flatten):
                pass
            elif isinstance(step, nn.Dropout):
                # Dropout is ignored at inference time -> identity
                pass
            else:
                # Explicitly reject known unsupported activations like Softmax
                if isinstance(step, nn.Softmax):
                    raise NoModel(predictor, "Softmax activation is not supported")
                raise NoModel(predictor, f"Unsupported layer {type(step).__name__}")
        super().__init__(gp_model, predictor, input_vars, output_vars)

    def _mip_model(self, **kwargs):
        network = self.predictor
        _input = self._input
        output = None
        num_layers = len(network)
        # If a Flatten follows spatial layers (Conv2d/MaxPool2d), remember the
        # pre-flatten NHWC shape so we can reorder the next Linear weights.
        pre_flat_spatial_shape = None

        for i, step in enumerate(network):
            if i == num_layers - 1:
                output = self._output
            if isinstance(step, nn.ReLU):
                layer = self._add_activation_layer(
                    _input, self.act_dict["relu"], output, name=f"relu_{i}", **kwargs
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
                # If we flattened a 4D NHWC tensor coming from spatial layers,
                # adjust Linear weights from PyTorch's (C,H,W) flatten order to
                # our NHWC row-major (H,W,C) order used in the MIP.
                if pre_flat_spatial_shape is not None:
                    H, W, C = (
                        pre_flat_spatial_shape[1],
                        pre_flat_spatial_shape[2],
                        pre_flat_spatial_shape[3],
                    )
                    N = H * W * C
                    if layer_weight.shape[0] == N:
                        pt_index_for_mip = [0] * N
                        for h in range(H):
                            for w in range(W):
                                for c in range(C):
                                    k_mip = h * (W * C) + w * C + c
                                    j_pt = c * (H * W) + h * W + w
                                    pt_index_for_mip[k_mip] = j_pt
                        layer_weight = layer_weight[np.array(pt_index_for_mip), :]
                    pre_flat_spatial_shape = None
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
            elif isinstance(step, nn.Conv2d):
                # Extract weights/bias and map to NHWC + (kh, kw, in_c, out_c)
                w = step.weight.detach().numpy()  # (out_c, in_c, kh, kw)
                b = (
                    step.bias.detach().numpy()
                    if step.bias is not None
                    else np.zeros((step.out_channels,), dtype=w.dtype)
                )
                # Convert to (kh, kw, in_c, out_c)
                w = np.transpose(w, (2, 3, 1, 0))

                # Normalize stride and padding
                stride = (
                    step.stride
                    if isinstance(step.stride, tuple)
                    else (step.stride, step.stride)
                )
                padding = step.padding
                if isinstance(padding, (tuple, list)):
                    pad_is_zero = all(p == 0 for p in padding)
                elif isinstance(padding, int):
                    pad_is_zero = padding == 0
                else:
                    # strings like 'same' are not supported
                    pad_is_zero = False

                kwargs["accepted_dim"] = (4,)
                layer = self._add_conv2d_layer(
                    _input,
                    w,
                    b,
                    step.out_channels,
                    step.kernel_size
                    if isinstance(step.kernel_size, tuple)
                    else (step.kernel_size, step.kernel_size),
                    stride,
                    "valid" if pad_is_zero else "unsupported",
                    self.act_dict["identity"],
                    output,
                    name=f"conv2d_{i}",
                    **kwargs,
                )
                _input = layer.output
            elif isinstance(step, nn.MaxPool2d):
                pool_size = (
                    step.kernel_size
                    if isinstance(step.kernel_size, tuple)
                    else (step.kernel_size, step.kernel_size)
                )
                stride = (
                    step.stride
                    if isinstance(step.stride, tuple)
                    else (step.stride if step.stride is not None else pool_size)
                )
                if not isinstance(stride, tuple):
                    stride = (stride, stride)
                padding = step.padding
                if isinstance(padding, (tuple, list)):
                    pad_is_zero = all(p == 0 for p in padding)
                elif isinstance(padding, int):
                    pad_is_zero = padding == 0
                else:
                    pad_is_zero = False
                kwargs["accepted_dim"] = (4,)
                layer = self._add_maxpool2d_layer(
                    _input,
                    pool_size,
                    stride,
                    "valid" if pad_is_zero else "unsupported",
                    output,
                    name=f"maxpool2d_{i}",
                    **kwargs,
                )
                _input = layer.output
            elif isinstance(step, nn.Flatten):
                kwargs["accepted_dim"] = (2,)
                # Record pre-flatten shape if the input is 4D NHWC
                pre_flat_spatial_shape = (
                    _input.shape if getattr(_input, "ndim", 0) == 4 else None
                )
                layer = self._add_flatten_layer(
                    _input,
                    output,
                    name=f"flatten_{i}",
                    **kwargs,
                )
                _input = layer.output
            elif isinstance(step, nn.Dropout):
                # Ignore dropout during inference
                layer = self._add_activation_layer(
                    _input,
                    self.act_dict["identity"],
                    output,
                    name=f"dropout_{i}",
                    **kwargs,
                )
                _input = layer.output
        if self._output is None:
            self._output = layer.output

    def get_error(self, eps=None):
        if self._has_solution:
            t_in = torch.from_numpy(self.input_values).float()
            # If the network contains Conv2d/MaxPool2d, expect PyTorch NCHW; convert from NHWC if needed
            has_spatial = any(
                isinstance(s, (nn.Conv2d, nn.MaxPool2d)) for s in self.predictor
            )
            if has_spatial and t_in.ndim == 4:
                # assume input is NHWC -> convert to NCHW
                t_in = t_in.permute(0, 3, 1, 2)
            t_out = self.predictor.forward(t_in)
            t_out_np = t_out.detach().numpy()
            # If output is 4D and came from spatial layers, convert back to NHWC for comparison
            if has_spatial and t_out_np.ndim == 4:
                t_out_np = np.transpose(t_out_np, (0, 2, 3, 1))
            r_val = np.abs(t_out_np - self.output_values)
            if eps is not None and np.max(r_val) > eps:
                print(f"{t_out} != {self.output_values}")
            return r_val
        raise NoSolution()
