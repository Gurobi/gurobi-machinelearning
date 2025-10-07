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

"""Module for formulating an ONNX MLP model into a :external+gurobi:py:class:`Model`.

Supported ONNX models are simple feed-forward networks composed of `Gemm`
nodes (dense layers) and `Relu` activations. This mirrors the Keras and
PyTorch integrations, which currently handle Dense/Linear + ReLU networks.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:  # Lazy import to avoid hard dependency when not used
    import onnx
    from onnx import numpy_helper
except Exception:  # pragma: no cover - handled by no-deps tests
    onnx = None  # type: ignore
    numpy_helper = None  # type: ignore

from ..exceptions import NoModel, NoSolution
from ..modeling.neuralnet import BaseNNConstr


def add_onnx_constr(gp_model, onnx_model, input_vars, output_vars=None, **kwargs):
    """Formulate an ONNX MLP model into `gp_model`.

    The formulation predicts the values of `output_vars` using `input_vars`
    according to `onnx_model`.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        Target Gurobi model where the predictor submodel is added.
    onnx_model : onnx.ModelProto
        ONNX model, expected to represent a sequential MLP with `Gemm` nodes
        and `Relu` activations.
    input_vars : mvar_array_like
        Decision variables used as input for the model in `gp_model`.
    output_vars : mvar_array_like, optional
        Decision variables used as output for the model in `gp_model`.

    Warnings
    --------
    Only networks composed of `Gemm` and `Relu` nodes are supported. `Gemm`
    nodes must use default `alpha=1`, `beta=1`. Attribute `transB` is
    supported.
    """
    return ONNXNetworkConstr(gp_model, onnx_model, input_vars, output_vars, **kwargs)


class _ONNXLayer:
    """Internal representation of one dense+activation block."""

    def __init__(self, W: np.ndarray, b: np.ndarray, activation: str = "identity"):
        self.W = W  # shape (in, out)
        self.b = b  # shape (out,)
        self.activation = activation  # "relu" or "identity"


class ONNXNetworkConstr(BaseNNConstr):
    """Formulate a supported ONNX MLP model as a Gurobi predictor constraint."""

    def __init__(self, gp_model, predictor, input_vars, output_vars=None, **kwargs):
        if onnx is None:
            raise NoModel(predictor, "onnx is not available")
        if not isinstance(predictor, onnx.ModelProto):
            raise NoModel(predictor, "Expected an onnx.ModelProto model")

        self._layers_spec: List[_ONNXLayer] = self._parse_mlp(predictor)
        if not self._layers_spec:
            raise NoModel(predictor, "Empty or unsupported ONNX graph")

        super().__init__(gp_model, predictor, input_vars, output_vars, **kwargs)

    def _parse_mlp(self, model: "onnx.ModelProto") -> List[_ONNXLayer]:
        """Parse a limited subset of ONNX graphs representing MLPs.

        We support sequences of: Gemm -> (Relu)? -> Gemm -> (Relu)? ...
        Gemm attributes allowed: alpha==1, beta==1, transB in {0,1}.
        """
        graph = model.graph

        # Map initializer name -> numpy array
        init_map = {}
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            init_map[init.name] = arr

        # Helper: get attribute value with default
        def _get_attr(node, name, default=None):
            for a in node.attribute:
                if a.name == name:
                    # attributes can be ints, floats
                    if a.type == onnx.AttributeProto.INT:
                        return int(a.i)
                    if a.type == onnx.AttributeProto.FLOAT:
                        return float(a.f)
            return default

        # Iterate nodes gathering dense layers and relus
        layers: List[_ONNXLayer] = []
        pending_activation: Optional[str] = None

        for node in graph.node:
            op = node.op_type
            if op == "Gemm":
                alpha = _get_attr(node, "alpha", 1.0)
                beta = _get_attr(node, "beta", 1.0)
                transB = _get_attr(node, "transB", 0)
                if alpha != 1.0 or beta != 1.0:
                    raise NoModel(
                        model, f"Unsupported Gemm attributes alpha={alpha}, beta={beta}"
                    )

                # Inputs: A, B, C
                if len(node.input) < 2:
                    raise NoModel(model, "Gemm node missing inputs")
                # B and C should be initializers
                B_name = node.input[1]
                C_name = node.input[2] if len(node.input) > 2 else None
                if B_name not in init_map:
                    raise NoModel(model, "Gemm weights must be an initializer")
                W = init_map[B_name]
                if transB == 1:
                    W = W.T  # make it shape (in, out)
                if C_name is None or C_name not in init_map:
                    b = np.zeros((W.shape[1],), dtype=W.dtype)
                else:
                    b = init_map[C_name].reshape(-1)
                    if b.shape[0] != W.shape[1]:
                        raise NoModel(model, "Gemm bias has wrong shape")

                act = pending_activation or "identity"
                layers.append(_ONNXLayer(W=W, b=b, activation=act))
                pending_activation = None

            elif op == "Relu":
                # Next linear layer will use relu activation; if we have no
                # preceding linear layer, we model it as a pure activation layer
                # via _add_activation_layer during _mip_model.
                # To keep the implementation simple and in line with Keras/Torch,
                # we treat Relu as activation for the preceding affine transform
                # when possible; otherwise mark pending_activation to apply to
                # the following Dense layer.
                if layers and layers[-1].activation == "identity":
                    layers[-1].activation = "relu"
                else:
                    # No prior dense, remember to insert a standalone activation
                    # at modeling time by setting a pending flag.
                    # Here we store a marker layer with zero-sized W to signal
                    # a standalone activation to the modeler.
                    layers.append(
                        _ONNXLayer(
                            W=np.zeros((0, 0)), b=np.zeros((0,)), activation="relu"
                        )
                    )
            elif op in ("Identity",):
                # Ignore
                continue
            else:
                raise NoModel(model, f"Unsupported ONNX op {op}")

        # Validate at least one real dense layer
        has_dense = any(layer.W.size > 0 for layer in layers)
        if not has_dense:
            return []
        return layers

    def _mip_model(self, **kwargs):
        _input = self._input
        output = None
        # Build Gurobi layers according to parsed spec
        for i, spec in enumerate(self._layers_spec):
            if i == len(self._layers_spec) - 1:
                output = self._output
            if spec.W.size == 0:
                # Standalone activation layer
                layer = self._add_activation_layer(
                    _input,
                    self.act_dict[spec.activation],
                    output,
                    name=f"relu{i}",
                    **kwargs,
                )
                _input = layer.output
            else:
                layer = self._add_dense_layer(
                    _input,
                    spec.W,
                    spec.b,
                    self.act_dict[
                        spec.activation if spec.activation != "identity" else "identity"
                    ],
                    output,
                    name=f"dense{i}",
                    **kwargs,
                )
                _input = layer.output
        if self._output is None:
            self._output = layer.output

    def _forward_numpy(self, X: np.ndarray) -> np.ndarray:
        out = X
        for spec in self._layers_spec:
            if spec.W.size == 0:
                # Relu activation directly on out
                out = np.maximum(out, 0.0)
            else:
                out = out @ spec.W + spec.b
                if spec.activation == "relu":
                    out = np.maximum(out, 0.0)
        return out

    def get_error(self, eps=None):
        if self._has_solution:
            pred = self._forward_numpy(self.input_values)
            r_val = np.abs(pred - self.output_values)
            if eps is not None and np.max(r_val) > eps:
                print(f"{pred} != {self.output_values}")
            return r_val
        raise NoSolution()
