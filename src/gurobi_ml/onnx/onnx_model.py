# Copyright © 2025 Gurobi Optimization, LLC
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

"""Module for formulating an ONNX model into a :external+gurobi:py:class:`Model`.

Supported ONNX models are sequential neural networks composed of:
- Dense layers: `Gemm` nodes or `MatMul`+`Add` sequences
- Convolutional layers: `Conv` nodes (2D convolution with valid padding)
- Pooling layers: `MaxPool` nodes (with valid padding)
- `Flatten` nodes
- `Relu` activations

This mirrors the Keras and PyTorch integrations for neural networks.
"""

from __future__ import annotations


import numpy as np
import onnx
from onnx import numpy_helper

from ..exceptions import NoModel, NoSolution
from ..modeling.neuralnet import BaseNNConstr


def add_onnx_constr(gp_model, onnx_model, input_vars, output_vars=None, **kwargs):
    """Formulate an ONNX neural network model into `gp_model`.

    The formulation predicts the values of `output_vars` using `input_vars`
    according to `onnx_model`.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        Target Gurobi model where the predictor submodel is added.
    onnx_model : onnx.ModelProto
        ONNX model representing a sequential neural network with supported
        operations (Gemm/MatMul+Add for dense layers, Conv for convolution,
        MaxPool for pooling, Flatten, and Relu activations).
    input_vars : mvar_array_like
        Decision variables used as input for the model in `gp_model`.
        For convolutional models, variables should be in NHWC format
        (batch, height, width, channels), even though ONNX uses NCHW format.
    output_vars : mvar_array_like, optional
        Decision variables used as output for the model in `gp_model`.

    Warnings
    --------
    Supported operations:
    - Dense layers: `Gemm` (with alpha=1, beta=1, transB in {0,1}) or `MatMul`+`Add`
    - Convolutional layers: `Conv` (2D, valid padding only)
    - Pooling: `MaxPool` (valid padding only)
    - `Flatten` (axis=1)
    - `Relu` activation

    Notes
    -----
    For models with convolutional layers, input_vars should use NHWC format
    (batch, height, width, channels), which differs from ONNX's NCHW format.
    The weight tensors are automatically converted during model parsing.
    """
    return ONNXNetworkConstr(gp_model, onnx_model, input_vars, output_vars, **kwargs)


class _ONNXLayer:
    """Internal representation of one layer (dense or conv2d or pooling/flatten)."""

    def __init__(
        self,
        layer_type: str,
        W: np.ndarray | None = None,
        b: np.ndarray | None = None,
        activation: str = "identity",
        channels: int | None = None,
        kernel_size: tuple[int, int] | None = None,
        stride: tuple[int, int] | None = None,
        padding: str = "valid",
        pool_size: tuple[int, int] | None = None,
    ):
        self.layer_type = layer_type  # "dense", "conv2d", "maxpool2d", "flatten"
        self.W = W  # For dense: (in, out); For conv2d: (kh, kw, in_c, out_c)
        self.b = b  # shape (out,) or (channels,)
        self.activation = activation  # "relu" or "identity"
        # Conv2d specific
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # MaxPool2d specific
        self.pool_size = pool_size


class ONNXNetworkConstr(BaseNNConstr):
    """Formulate a supported ONNX neural network model as a Gurobi predictor constraint."""

    def __init__(self, gp_model, predictor, input_vars, output_vars=None, **kwargs):
        if not isinstance(predictor, onnx.ModelProto):
            raise NoModel(predictor, "Expected an onnx.ModelProto model")

        self._layers_spec: list[_ONNXLayer] = self._parse_mlp(predictor)
        if not self._layers_spec:
            raise NoModel(predictor, "Empty or unsupported ONNX graph")

        super().__init__(gp_model, predictor, input_vars, output_vars, **kwargs)

    def _validate_sequential_architecture(self, graph, init_map):
        """Validate that the graph has a sequential architecture.

        Raises NoModel if the graph contains:
        - Skip connections (same intermediate value used by multiple nodes)
        - Residual connections (Add nodes combining non-bias values)
        - Non-sequential topology
        """
        # Build usage map: which nodes use each tensor
        tensor_usage = {}
        for node in graph.node:
            for inp in node.input:
                if inp not in tensor_usage:
                    tensor_usage[inp] = []
                tensor_usage[inp].append(node.name)

        # Check 1: Input should only be used by one node (first layer)
        for graph_input in graph.input:
            input_name = graph_input.name
            if input_name in tensor_usage and len(tensor_usage[input_name]) > 1:
                raise NoModel(
                    graph,
                    f"Non-sequential architecture detected: input '{input_name}' is used by multiple nodes {tensor_usage[input_name]}. "
                    "Skip connections and residual architectures are not supported.",
                )

        # Check 2: Each intermediate node output should be used by at most one node
        # (except for the final output which may not be used by any node)
        for node in graph.node:
            for output in node.output:
                if output in tensor_usage and len(tensor_usage[output]) > 1:
                    raise NoModel(
                        graph,
                        f"Non-sequential architecture detected: node '{node.name}' output '{output}' is used by multiple nodes {tensor_usage[output]}. "
                        "Skip connections and residual architectures are not supported.",
                    )

        # Check 3: Add nodes should only be used for bias addition (MatMul+Add pattern)
        # Not for combining two computed branches (residual connections)
        for node in graph.node:
            if node.op_type == "Add":
                # An Add is valid if one of its inputs is an initializer (bias)
                # and the other is from a MatMul
                inputs = list(node.input)
                if len(inputs) != 2:
                    continue

                # Check if this is a MatMul+Add pattern (one input from MatMul, one is initializer)
                is_bias_add = False
                for inp in inputs:
                    if inp in init_map:
                        # One input is a constant (bias)
                        is_bias_add = True
                        break

                if not is_bias_add:
                    # Both inputs are computed values - this is a residual connection
                    raise NoModel(
                        graph,
                        f"Non-sequential architecture detected: Add node '{node.name}' combines two computed values {inputs}. "
                        "Residual connections are not supported.",
                    )

    def _parse_mlp(self, model: onnx.ModelProto) -> list[_ONNXLayer]:
        """Parse ONNX graphs representing sequential neural networks.

        We support sequences of:
        - Gemm and MatMul+Add nodes (dense layers)
        - Conv nodes (2D convolution)
        - MaxPool nodes (2D max pooling)
        - Flatten nodes
        - Relu activations

        Gemm attributes allowed: alpha==1, beta==1, transB in {0,1}.
        Conv and MaxPool only support valid padding (pads=0).
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
                    # attributes can be ints, floats, or lists of ints
                    if a.type == onnx.AttributeProto.INT:
                        return int(a.i)
                    if a.type == onnx.AttributeProto.FLOAT:
                        return float(a.f)
                    if a.type == onnx.AttributeProto.INTS:
                        return list(a.ints)
            return default

        # Validate that the graph is sequential (no skip connections or residual adds)
        self._validate_sequential_architecture(graph, init_map)

        # Build a map from output name to node for easier traversal
        output_to_node = {}
        for node in graph.node:
            for output in node.output:
                output_to_node[output] = node

        # Iterate nodes gathering dense layers and relus
        layers: list[_ONNXLayer] = []
        pending_activation: str | None = None
        processed_indices = set()

        for node_idx, node in enumerate(graph.node):
            if node_idx in processed_indices:
                continue

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
                layers.append(_ONNXLayer(layer_type="dense", W=W, b=b, activation=act))
                pending_activation = None
                processed_indices.add(node_idx)

            elif op == "MatMul":
                # MatMul should be followed by Add for bias
                # Inputs: A, B where B is the weight matrix
                if len(node.input) != 2:
                    raise NoModel(model, "MatMul node should have exactly 2 inputs")

                weight_name = node.input[1]
                if weight_name not in init_map:
                    raise NoModel(model, "MatMul weights must be an initializer")

                W = init_map[weight_name]  # shape should be (in, out)

                # Look for an Add node that uses the output of this MatMul
                matmul_output = node.output[0]
                add_node = None
                add_node_idx = None
                for next_idx, next_node in enumerate(graph.node):
                    if next_node.op_type == "Add" and matmul_output in next_node.input:
                        add_node = next_node
                        add_node_idx = next_idx
                        break

                if add_node is not None:
                    # Find the bias input (the one that's not the MatMul output)
                    bias_name = None
                    for inp in add_node.input:
                        if inp != matmul_output and inp in init_map:
                            bias_name = inp
                            break

                    if bias_name is not None:
                        b = init_map[bias_name].reshape(-1)
                        if b.shape[0] != W.shape[1]:
                            raise NoModel(model, "Add bias has wrong shape")
                    else:
                        b = np.zeros((W.shape[1],), dtype=W.dtype)

                    processed_indices.add(add_node_idx)
                else:
                    # MatMul without Add - use zero bias
                    b = np.zeros((W.shape[1],), dtype=W.dtype)

                act = pending_activation or "identity"
                layers.append(_ONNXLayer(layer_type="dense", W=W, b=b, activation=act))
                pending_activation = None
                processed_indices.add(node_idx)

            elif op == "Conv":
                # ONNX Conv node for 2D convolution
                # Inputs: X (NCHW), W (out_channels, in_channels, kH, kW), [B]
                if len(node.input) < 2:
                    raise NoModel(model, "Conv node missing inputs")

                weight_name = node.input[1]
                if weight_name not in init_map:
                    raise NoModel(model, "Conv weights must be an initializer")

                W = init_map[weight_name]  # shape: (out_c, in_c, kh, kw)

                # Get bias if present
                if len(node.input) > 2:
                    bias_name = node.input[2]
                    if bias_name in init_map:
                        b = init_map[bias_name].reshape(-1)
                    else:
                        b = np.zeros((W.shape[0],), dtype=W.dtype)
                else:
                    b = np.zeros((W.shape[0],), dtype=W.dtype)

                # Extract Conv attributes
                kernel_shape = _get_attr(node, "kernel_shape", None)
                if kernel_shape is None:
                    kernel_shape = (W.shape[2], W.shape[3])
                else:
                    kernel_shape = tuple(kernel_shape)

                strides = _get_attr(node, "strides", [1, 1])
                if isinstance(strides, int):
                    strides = (strides, strides)
                else:
                    strides = tuple(strides)

                pads = _get_attr(node, "pads", [0, 0, 0, 0])
                # pads is [x1_begin, x2_begin, x1_end, x2_end] in ONNX
                # We only support valid padding (all zeros)
                if isinstance(pads, (list, tuple)):
                    pad_is_zero = all(p == 0 for p in pads)
                else:
                    pad_is_zero = pads == 0

                if not pad_is_zero:
                    raise NoModel(model, "Conv with non-zero padding is not supported")

                # Convert ONNX weight format (out_c, in_c, kh, kw) to our format (kh, kw, in_c, out_c)
                W = np.transpose(W, (2, 3, 1, 0))

                out_channels = W.shape[3]
                act = pending_activation or "identity"
                layers.append(
                    _ONNXLayer(
                        layer_type="conv2d",
                        W=W,
                        b=b,
                        activation=act,
                        channels=out_channels,
                        kernel_size=kernel_shape,
                        stride=strides,
                        padding="valid",
                    )
                )
                pending_activation = None
                processed_indices.add(node_idx)

            elif op == "MaxPool":
                # MaxPool2d operation
                kernel_shape = _get_attr(node, "kernel_shape", None)
                if kernel_shape is None:
                    raise NoModel(model, "MaxPool requires kernel_shape attribute")
                kernel_shape = tuple(kernel_shape)

                strides = _get_attr(node, "strides", kernel_shape)
                if isinstance(strides, int):
                    strides = (strides, strides)
                else:
                    strides = tuple(strides)

                pads = _get_attr(node, "pads", [0, 0, 0, 0])
                if isinstance(pads, (list, tuple)):
                    pad_is_zero = all(p == 0 for p in pads)
                else:
                    pad_is_zero = pads == 0

                if not pad_is_zero:
                    raise NoModel(
                        model, "MaxPool with non-zero padding is not supported"
                    )

                layers.append(
                    _ONNXLayer(
                        layer_type="maxpool2d",
                        pool_size=kernel_shape,
                        stride=strides,
                        padding="valid",
                        activation="identity",
                    )
                )
                processed_indices.add(node_idx)

            elif op == "Flatten":
                # Flatten operation - axis parameter determines how to reshape
                # Default axis=1 means flatten from dimension 1 onwards
                axis = _get_attr(node, "axis", 1)
                if axis != 1:
                    raise NoModel(
                        model,
                        f"Flatten with axis={axis} is not supported (only axis=1)",
                    )

                layers.append(_ONNXLayer(layer_type="flatten", activation="identity"))
                processed_indices.add(node_idx)

            elif op == "Add":
                # Skip if already processed as part of MatMul+Add
                if node_idx not in processed_indices:
                    # Standalone Add node - ignore or warn?
                    processed_indices.add(node_idx)

            elif op == "Relu":
                # Next linear/conv layer will use relu activation; if we have no
                # preceding layer, we model it as a pure activation layer
                # via _add_activation_layer during _mip_model.
                if layers and layers[-1].activation == "identity":
                    layers[-1].activation = "relu"
                else:
                    # No prior layer, store a standalone activation marker
                    layers.append(
                        _ONNXLayer(layer_type="activation", activation="relu")
                    )
                processed_indices.add(node_idx)

            elif op in ("Identity",):
                # Ignore
                processed_indices.add(node_idx)
                continue
            else:
                raise NoModel(model, f"Unsupported ONNX op {op}")

        # Validate at least one real layer (dense or conv)
        has_layer = any(layer.layer_type in ("dense", "conv2d") for layer in layers)
        if not has_layer:
            return []

        # Post-process: if we have Conv -> ... -> Flatten -> Dense,
        # we need to reorder the Dense weights because ONNX uses NCHW flattening
        # but our internal representation uses NHWC flattening
        self._fix_dense_weights_after_flatten(layers, graph)

        return layers

    def _fix_dense_weights_after_flatten(self, layers, graph):
        """Fix Dense layer weights that follow Flatten after Conv layers.

        ONNX models flatten in NCHW order, but our Conv2DLayer outputs NHWC.
        So we need to reorder the Dense weights to match NHWC flattening.
        """
        # Find patterns: Conv/MaxPool -> ... -> Flatten -> Dense
        for i in range(len(layers) - 1):
            if layers[i].layer_type == "flatten":
                # Check if there's a conv/maxpool before flatten
                has_conv_before = any(
                    layers[j].layer_type in ("conv2d", "maxpool2d") for j in range(i)
                )
                # Check if there's a dense layer right after flatten
                if (
                    i + 1 < len(layers)
                    and layers[i + 1].layer_type == "dense"
                    and has_conv_before
                ):
                    # Need to find the shape before flattening
                    # This requires tracking through the layers
                    shape_before_flatten = self._compute_shape_before_flatten(
                        layers, i, graph
                    )
                    if (
                        shape_before_flatten is not None
                        and len(shape_before_flatten) == 4
                    ):
                        batch, height, width, channels = shape_before_flatten
                        # Reorder dense weights from NCHW-flattened to NHWC-flattened
                        layers[i + 1].W = self._reorder_dense_weights_nchw_to_nhwc(
                            layers[i + 1].W, height, width, channels
                        )

    def _compute_shape_before_flatten(self, layers, flatten_idx, graph):
        """Compute the shape of data just before the Flatten layer."""
        # We need to track through layers to compute shapes
        # For now, use a simplified approach: get from input shape and layer specs
        # Start from input shape (which we'll get from graph)
        input_shape = None
        for inp in graph.input:
            input_shape = [
                d.dim_value if d.dim_value > 0 else 1
                for d in inp.type.tensor_type.shape.dim
            ]
            break

        if input_shape is None:
            return None

        # Convert from NCHW (ONNX) to NHWC (our format)
        if len(input_shape) == 4:
            batch, channels, height, width = input_shape
            current_shape = [batch, height, width, channels]
        else:
            current_shape = input_shape

        # Simulate forward pass through layers
        for i in range(flatten_idx):
            layer = layers[i]
            if layer.layer_type == "conv2d":
                batch, height, width, in_channels = current_shape
                kh, kw = layer.kernel_size
                sh, sw = layer.stride
                # Output shape after conv (valid padding)
                out_h = (height - kh) // sh + 1
                out_w = (width - kw) // sw + 1
                current_shape = [batch, out_h, out_w, layer.channels]
            elif layer.layer_type == "maxpool2d":
                batch, height, width, channels = current_shape
                ph, pw = layer.pool_size
                sh, sw = layer.stride
                out_h = (height - ph) // sh + 1
                out_w = (width - pw) // sw + 1
                current_shape = [batch, out_h, out_w, channels]
            # activation layers don't change shape

        return current_shape

    def _reorder_dense_weights_nchw_to_nhwc(self, W, height, width, channels):
        """Reorder dense layer weights from NCHW-flattened input to NHWC-flattened input.

        W has shape (flat_size, out_features) where flat_size = height * width * channels
        ONNX flattens as: [c0_h0_w0, c0_h0_w1, ..., c0_hH_wW, c1_h0_w0, ..., cC_hH_wW]
        We flatten as:    [h0_w0_c0, h0_w0_c1, ..., h0_w0_cC, h0_w1_c0, ..., hH_wW_cC]

        We need to reorder the input dimension of W accordingly.
        """
        flat_size, out_features = W.shape
        expected_size = height * width * channels
        if flat_size != expected_size:
            # Shape mismatch, can't reorder safely
            return W

        # Create index mapping from NCHW to NHWC
        W_reordered = np.zeros_like(W)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    # NCHW index: channels vary slowest
                    nchw_idx = c * (height * width) + h * width + w
                    # NHWC index: channels vary fastest
                    nhwc_idx = h * (width * channels) + w * channels + c
                    # Copy the row
                    W_reordered[nhwc_idx, :] = W[nchw_idx, :]

        return W_reordered

    def _mip_model(self, **kwargs):
        _input = self._input
        output = None
        # Build Gurobi layers according to parsed spec
        for i, spec in enumerate(self._layers_spec):
            if i == len(self._layers_spec) - 1:
                output = self._output

            if spec.layer_type == "activation":
                # Standalone activation layer
                layer = self._add_activation_layer(
                    _input,
                    self.act_dict[spec.activation],
                    output,
                    name=f"relu{i}",
                    **kwargs,
                )
                _input = layer.output
            elif spec.layer_type == "dense":
                layer = self._add_dense_layer(
                    _input,
                    spec.W,
                    spec.b,
                    self.act_dict[spec.activation],
                    output,
                    name=f"dense{i}",
                    **kwargs,
                )
                _input = layer.output
            elif spec.layer_type == "conv2d":
                kwargs["accepted_dim"] = (4,)
                layer = self._add_conv2d_layer(
                    _input,
                    spec.W,
                    spec.b,
                    spec.channels,
                    spec.kernel_size,
                    spec.stride,
                    spec.padding,
                    self.act_dict[spec.activation],
                    output,
                    name=f"conv2d{i}",
                    **kwargs,
                )
                _input = layer.output
            elif spec.layer_type == "maxpool2d":
                kwargs["accepted_dim"] = (4,)
                layer = self._add_maxpool2d_layer(
                    _input,
                    spec.pool_size,
                    spec.stride,
                    spec.padding,
                    output,
                    name=f"maxpool2d{i}",
                    **kwargs,
                )
                _input = layer.output
            elif spec.layer_type == "flatten":
                kwargs["accepted_dim"] = (2,)
                layer = self._add_flatten_layer(
                    _input,
                    output,
                    name=f"flatten{i}",
                    **kwargs,
                )
                _input = layer.output

        if self._output is None:
            self._output = layer.output

    def get_error(self, eps=None):
        if self._has_solution:
            import onnxruntime as ort

            # Check if model has convolutional layers
            has_conv = any(layer.layer_type == "conv2d" for layer in self._layers_spec)

            sess = ort.InferenceSession(self.predictor.SerializeToString())
            input_name = sess.get_inputs()[0].name

            # If model has Conv layers, input_values are in NHWC format
            # but ONNX expects NCHW, so we need to convert
            input_data = self.input_values.astype(np.float32)
            if has_conv and input_data.ndim == 4:
                # Convert from NHWC to NCHW for ONNX inference
                input_data = np.transpose(input_data, (0, 3, 1, 2))

            pred = sess.run(None, {input_name: input_data})[0]

            # If output is 4D and model has conv layers, convert back from NCHW to NHWC
            if has_conv and pred.ndim == 4:
                pred = np.transpose(pred, (0, 2, 3, 1))

            r_val = np.abs(pred - self.output_values)
            if eps is not None and np.max(r_val) > eps:
                print(f"{pred} != {self.output_values}")
            return r_val
        raise NoSolution()
