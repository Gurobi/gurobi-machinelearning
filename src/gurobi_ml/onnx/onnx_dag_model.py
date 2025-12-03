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

"""Module for formulating ONNX models with DAG topology into a Gurobi Model.

This module supports ONNX models with arbitrary graph topologies, including:
- Skip connections
- Residual connections
- Multi-branch architectures

Supported ONNX operations:
- Gemm (dense layers)
- MatMul + Add (dense layers)
- Conv (2D convolutional layers)
- MaxPool (2D max pooling)
- Flatten (flatten operation)
- BatchNormalization (batch normalization)
- Relu (activation)
- Add (element-wise addition for residual connections)
- Identity (pass-through)
"""

from __future__ import annotations

import graphlib

import numpy as np
import onnx
from onnx import numpy_helper

from ..exceptions import NoModel, NoSolution
from ..modeling.neuralnet import DAGNNConstr


def add_onnx_dag_constr(gp_model, onnx_model, input_vars, output_vars=None, **kwargs):
    """Formulate an ONNX model with DAG topology into `gp_model`.

    The formulation predicts the values of `output_vars` using `input_vars`
    according to `onnx_model`. This function supports models with skip connections,
    residual connections, and other non-sequential topologies.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        Target Gurobi model where the predictor submodel is added.
    onnx_model : onnx.ModelProto
        ONNX model with potentially non-sequential topology.
    input_vars : mvar_array_like
        Decision variables used as input for the model in `gp_model`.
    output_vars : mvar_array_like, optional
        Decision variables used as output for the model in `gp_model`.

    Returns
    -------
    ONNXDAGNetworkConstr
        Object managing the Gurobi formulation of the ONNX model

    Notes
    -----
    Supported operations:
    - Gemm (with alpha=1, beta=1, transB=0 or 1)
    - MatMul
    - Add (for both bias addition and residual connections)
    - Conv (2D convolution with symmetric padding)
    - MaxPool (2D max pooling with symmetric padding)
    - Flatten (axis=1)
    - BatchNormalization (all parameters must be initializers)
    - Relu
    - Identity

    The implementation handles arbitrary DAG topologies by:
    1. Performing topological sort of the computation graph
    2. Processing nodes in dependency order
    3. Tracking tensor variables through the graph
    4. Supporting tensor reuse (skip/residual connections)
    """
    return ONNXDAGNetworkConstr(gp_model, onnx_model, input_vars, output_vars, **kwargs)


class ONNXDAGNetworkConstr(DAGNNConstr):
    """Formulate an ONNX model with DAG topology as a Gurobi predictor constraint.

    This class extends DAGNNConstr to support ONNX models with arbitrary graph
    topologies, including skip connections and residual connections.
    """

    def __init__(self, gp_model, predictor, input_vars, output_vars=None, **kwargs):
        if not isinstance(predictor, onnx.ModelProto):
            raise NoModel(predictor, "Expected an onnx.ModelProto model")

        # Will be set during parsing
        self._onnx_graph = None
        self._init_map = {}
        self._node_output_shapes = {}
        self._tensor_is_spatial = {}  # Track which tensors have spatial dims (from Conv/MaxPool)

        super().__init__(gp_model, predictor, input_vars, output_vars, **kwargs)

    def _mip_model(self, **kwargs):
        """Build the MIP model for the ONNX graph."""
        graph = self.predictor.graph
        self._onnx_graph = graph

        # Build initializer map
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            self._init_map[init.name] = arr

        # Register input tensor
        # Ensure input has batch dimension (reshape 1D to 2D if needed)
        input_name = graph.input[0].name
        input_vars = self._input
        if input_vars.ndim == 1:
            input_vars = input_vars.reshape((1, -1))
        self.set_tensor_vars(input_name, input_vars)

        # Topologically sort nodes
        sorted_nodes = self._topological_sort(graph)

        # Process each node in order
        for node in sorted_nodes:
            self._process_node(node, **kwargs)

        # Set output
        output_name = graph.output[0].name
        output_vars = self.get_tensor_vars(output_name)

        # If input was 1D (no batch dimension), reshape output back to 1D
        if (
            self._input.ndim == 1
            and output_vars.ndim == 2
            and output_vars.shape[0] == 1
        ):
            output_vars = output_vars.reshape(-1)

        if self._output is None:
            self._output = output_vars
        else:
            # Constrain provided output vars to match computed output
            self.gp_model.addConstr(
                self._output == output_vars, name="output_constraint"
            )

    def _topological_sort(self, graph):
        """Perform topological sort of the graph nodes using Python's graphlib.

        Parameters
        ----------
        graph : onnx.GraphProto
            The ONNX computation graph

        Returns
        -------
        list of onnx.NodeProto
            Nodes in topological order

        Raises
        ------
        NoModel
            If the graph contains cycles (not a valid DAG)
        """
        # Build node list and index mapping
        node_list = list(graph.node)

        # Map output tensor name to node index
        outputs_to_idx = {}
        for i, node in enumerate(node_list):
            for out in node.output:
                outputs_to_idx[out] = i

        # Use Python's standard library TopologicalSorter
        ts = graphlib.TopologicalSorter()

        # Add nodes and their dependencies
        for i, node in enumerate(node_list):
            # Find which nodes this node depends on
            predecessors = []
            for inp in node.input:
                # Skip initializers and graph inputs
                if inp in self._init_map or inp in {gi.name for gi in graph.input}:
                    continue
                # This input must be an output of another node
                if inp in outputs_to_idx:
                    predecessors.append(outputs_to_idx[inp])

            # Add node with its dependencies
            ts.add(i, *predecessors)

        # Get topological order
        try:
            # static_order() returns an iterable in topological order
            sorted_indices = list(ts.static_order())
        except graphlib.CycleError as e:
            raise NoModel(graph, f"Graph contains cycles - not a valid DAG: {e}")

        # Convert indices back to node objects
        return [node_list[i] for i in sorted_indices]

    def _process_node(self, node, **kwargs):
        """Process a single ONNX node.

        Parameters
        ----------
        node : onnx.NodeProto
            The node to process
        """
        op_type = node.op_type

        if op_type == "Gemm":
            self._process_gemm(node, **kwargs)
        elif op_type == "MatMul":
            self._process_matmul(node, **kwargs)
        elif op_type == "Add":
            self._process_add(node, **kwargs)
        elif op_type == "Relu":
            self._process_relu(node, **kwargs)
        elif op_type == "Identity":
            self._process_identity(node, **kwargs)
        elif op_type == "Conv":
            self._process_conv(node, **kwargs)
        elif op_type == "MaxPool":
            self._process_maxpool(node, **kwargs)
        elif op_type == "Flatten":
            self._process_flatten(node, **kwargs)
        elif op_type == "BatchNormalization":
            self._process_batchnorm(node, **kwargs)
        elif op_type == "Dropout":
            self._process_dropout(node, **kwargs)
        else:
            raise NoModel(self._onnx_graph, f"Unsupported ONNX op: {op_type}")

    def _get_attr(self, node, name, default=None):
        """Get attribute value from node."""
        for a in node.attribute:
            if a.name == name:
                if a.type == onnx.AttributeProto.INT:
                    return int(a.i)
                if a.type == onnx.AttributeProto.FLOAT:
                    return float(a.f)
                if a.type == onnx.AttributeProto.INTS:
                    return list(a.ints)
        return default

    def _reorder_weights_if_needed(self, input_tensor_name, W):
        """Reorder dense layer weights if input comes from flattened spatial data.

        ONNX flattens spatial data (from Conv/MaxPool) in NCHW order, but our
        internal representation uses NHWC order. This requires reordering the
        dense layer weights accordingly.

        Parameters
        ----------
        input_tensor_name : str
            Name of the input tensor to the dense layer
        W : np.ndarray
            Weight matrix with shape (in_features, out_features)

        Returns
        -------
        np.ndarray
            Reordered weight matrix (if needed) or original W
        """
        if input_tensor_name not in self._tensor_is_spatial:
            return W

        spatial_shape = self._tensor_is_spatial[input_tensor_name]
        if not isinstance(spatial_shape, tuple):
            return W

        # spatial_shape is (batch, height, width, channels) in NHWC format
        if len(spatial_shape) != 4:
            return W

        batch, height, width, channels = spatial_shape
        flat_size = height * width * channels

        if W.shape[0] != flat_size:
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

    def _process_gemm(self, node, **kwargs):
        """Process a Gemm (dense layer) node.

        Gemm computes: Y = alpha * A @ B + beta * C
        We support alpha=1, beta=1, with optional transB.
        """
        alpha = self._get_attr(node, "alpha", 1.0)
        beta = self._get_attr(node, "beta", 1.0)
        transB = self._get_attr(node, "transB", 0)

        if alpha != 1.0 or beta != 1.0:
            raise NoModel(
                self._onnx_graph,
                f"Unsupported Gemm attributes alpha={alpha}, beta={beta}",
            )

        # Get inputs: A (data), B (weight), C (bias)
        input_name = node.input[0]
        weight_name = node.input[1]
        bias_name = node.input[2] if len(node.input) > 2 else None

        input_vars = self.get_tensor_vars(input_name)
        W = self._init_map[weight_name]
        if transB == 1:
            W = W.T

        # Check if we need to reorder weights (flatten from spatial data)
        W = self._reorder_weights_if_needed(input_name, W)

        if bias_name and bias_name in self._init_map:
            b = self._init_map[bias_name].reshape(-1)
        else:
            b = np.zeros((W.shape[1],), dtype=W.dtype)

        # Create dense layer with identity activation
        # Activation will be fused if next node is Relu
        layer = self._add_dense_layer(
            input_vars,
            W,
            b,
            self.act_dict["identity"],
            name=node.name,
            **kwargs,
        )

        # Register output
        output_name = node.output[0]
        self.set_tensor_vars(output_name, layer.output)

    def _process_matmul(self, node, **kwargs):
        """Process a MatMul node.

        MatMul computes: Y = A @ B
        Usually followed by an Add for bias.
        """
        input_name = node.input[0]
        weight_name = node.input[1]

        input_vars = self.get_tensor_vars(input_name)
        W = self._init_map[weight_name]

        # Check if we need to reorder weights (flatten from spatial data)
        W = self._reorder_weights_if_needed(input_name, W)

        # MatMul without bias (bias will be added by subsequent Add node if present)
        b = np.zeros((W.shape[1],), dtype=W.dtype)

        layer = self._add_dense_layer(
            input_vars,
            W,
            b,
            self.act_dict["identity"],
            name=node.name,
            **kwargs,
        )

        output_name = node.output[0]
        self.set_tensor_vars(output_name, layer.output)

    def _process_add(self, node, **kwargs):
        """Process an Add node.

        Add can be:
        1. Bias addition (MatMul result + bias constant)
        2. Residual connection (two computed tensors)
        """
        input1_name = node.input[0]
        input2_name = node.input[1]

        # Check if either input is a constant (initializer)
        input1_is_const = input1_name in self._init_map
        input2_is_const = input2_name in self._init_map

        if input1_is_const or input2_is_const:
            # Bias addition: one input is constant, other is computed
            if input1_is_const:
                # Y = X + bias
                computed_vars = self.get_tensor_vars(input2_name)
                bias = self._init_map[input1_name].reshape(-1)
            else:
                # Y = X + bias
                computed_vars = self.get_tensor_vars(input1_name)
                bias = self._init_map[input2_name].reshape(-1)

            # For bias addition, we can fold it into the previous layer if it was identity
            # For now, create a simple constraint
            output_vars = self.gp_model.addMVar(
                computed_vars.shape,
                lb=-self.gp_model.Params.Infinity,
                name=f"{node.name}_out",
            )
            self.gp_model.addConstr(
                output_vars == computed_vars + bias, name=f"{node.name}_bias"
            )
        else:
            # Residual connection: both inputs are computed tensors
            input1_vars = self.get_tensor_vars(input1_name)
            input2_vars = self.get_tensor_vars(input2_name)

            # Create addition layer
            layer = self._add_add_layer(
                [input1_vars, input2_vars],
                name=node.name,
                **kwargs,
            )
            output_vars = layer.output

        output_name = node.output[0]
        self.set_tensor_vars(output_name, output_vars)

    def _process_relu(self, node, **kwargs):
        """Process a Relu activation node."""
        input_name = node.input[0]
        input_vars = self.get_tensor_vars(input_name)

        layer = self._add_activation_layer(
            input_vars,
            self.act_dict["relu"],
            name=node.name,
            **kwargs,
        )

        output_name = node.output[0]
        self.set_tensor_vars(output_name, layer.output)

    def _process_identity(self, node, **kwargs):
        """Process an Identity node (pass-through)."""
        input_name = node.input[0]
        input_vars = self.get_tensor_vars(input_name)

        output_name = node.output[0]
        self.set_tensor_vars(output_name, input_vars)

    def _process_conv(self, node, **kwargs):
        """Process a Conv (2D convolution) node.

        Conv computes 2D convolution with optional padding.
        ONNX uses NCHW format, but our internal representation uses NHWC.
        """
        # Get inputs: X (NCHW), W (out_channels, in_channels, kH, kW), [B]
        input_name = node.input[0]
        weight_name = node.input[1]

        input_vars = self.get_tensor_vars(input_name)
        W = self._init_map[weight_name]  # shape: (out_c, in_c, kh, kw)

        # Get bias if present
        if len(node.input) > 2:
            bias_name = node.input[2]
            if bias_name in self._init_map:
                b = self._init_map[bias_name].reshape(-1)
            else:
                b = np.zeros((W.shape[0],), dtype=W.dtype)
        else:
            b = np.zeros((W.shape[0],), dtype=W.dtype)

        # Extract Conv attributes
        kernel_shape = self._get_attr(node, "kernel_shape", None)
        if kernel_shape is None:
            kernel_shape = (W.shape[2], W.shape[3])
        else:
            kernel_shape = tuple(kernel_shape)

        strides = self._get_attr(node, "strides", [1, 1])
        if isinstance(strides, int):
            strides = (strides, strides)
        else:
            strides = tuple(strides)

        pads = self._get_attr(node, "pads", [0, 0, 0, 0])
        # pads is [x1_begin, x2_begin, x1_end, x2_end] in ONNX
        # For symmetric padding, x1_begin should equal x1_end, x2_begin should equal x2_end
        if isinstance(pads, (list, tuple)) and len(pads) == 4:
            # Check if padding is symmetric
            if pads[0] == pads[2] and pads[1] == pads[3]:
                # Symmetric padding - use (pad_h, pad_w)
                padding_tuple = (pads[0], pads[1])
            else:
                raise NoModel(
                    self._onnx_graph,
                    f"Conv with asymmetric padding {pads} is not supported. "
                    "Only symmetric padding is supported.",
                )
        elif isinstance(pads, int):
            padding_tuple = (pads, pads)
        else:
            padding_tuple = (0, 0)

        # Use tuple format or "valid" string
        if padding_tuple == (0, 0):
            padding = "valid"
        else:
            padding = padding_tuple

        # Convert ONNX weight format (out_c, in_c, kh, kw) to our format (kh, kw, in_c, out_c)
        W = np.transpose(W, (2, 3, 1, 0))

        out_channels = W.shape[3]

        # Create conv2d layer with identity activation
        # Activation will be fused if next node is Relu
        kwargs_copy = kwargs.copy()
        kwargs_copy["accepted_dim"] = (4,)
        layer = self._add_conv2d_layer(
            input_vars,
            W,
            b,
            out_channels,
            kernel_shape,
            strides,
            padding,
            self.act_dict["identity"],
            name=node.name,
            **kwargs_copy,
        )

        # Register output
        output_name = node.output[0]
        self.set_tensor_vars(output_name, layer.output)
        # Mark this tensor as spatial (4D from Conv)
        self._tensor_is_spatial[output_name] = True

    def _process_maxpool(self, node, **kwargs):
        """Process a MaxPool (2D max pooling) node."""
        input_name = node.input[0]
        input_vars = self.get_tensor_vars(input_name)

        # Extract MaxPool attributes
        kernel_shape = self._get_attr(node, "kernel_shape", None)
        if kernel_shape is None:
            raise NoModel(self._onnx_graph, "MaxPool requires kernel_shape attribute")
        kernel_shape = tuple(kernel_shape)

        strides = self._get_attr(node, "strides", kernel_shape)
        if isinstance(strides, int):
            strides = (strides, strides)
        else:
            strides = tuple(strides)

        pads = self._get_attr(node, "pads", [0, 0, 0, 0])
        # Parse padding similar to Conv
        if isinstance(pads, (list, tuple)) and len(pads) == 4:
            # Check if padding is symmetric
            if pads[0] == pads[2] and pads[1] == pads[3]:
                padding_tuple = (pads[0], pads[1])
            else:
                raise NoModel(
                    self._onnx_graph,
                    f"MaxPool with asymmetric padding {pads} is not supported. "
                    "Only symmetric padding is supported.",
                )
        elif isinstance(pads, int):
            padding_tuple = (pads, pads)
        else:
            padding_tuple = (0, 0)

        if padding_tuple == (0, 0):
            padding = "valid"
        else:
            padding = padding_tuple

        kwargs_copy = kwargs.copy()
        kwargs_copy["accepted_dim"] = (4,)
        layer = self._add_maxpool2d_layer(
            input_vars,
            kernel_shape,
            strides,
            padding,
            name=node.name,
            **kwargs_copy,
        )

        output_name = node.output[0]
        self.set_tensor_vars(output_name, layer.output)
        # Mark this tensor as spatial (4D from MaxPool)
        self._tensor_is_spatial[output_name] = True

    def _process_flatten(self, node, **kwargs):
        """Process a Flatten node."""
        input_name = node.input[0]
        input_vars = self.get_tensor_vars(input_name)

        # Default axis=1 means flatten from dimension 1 onwards
        axis = self._get_attr(node, "axis", 1)
        if axis != 1:
            raise NoModel(
                self._onnx_graph,
                f"Flatten with axis={axis} is not supported (only axis=1)",
            )

        kwargs_copy = kwargs.copy()
        kwargs_copy["accepted_dim"] = (2,)
        layer = self._add_flatten_layer(
            input_vars,
            name=node.name,
            **kwargs_copy,
        )

        output_name = node.output[0]
        self.set_tensor_vars(output_name, layer.output)

        # If input was spatial, mark the flatten output as coming from spatial
        # This will be used to reorder weights in subsequent dense layers
        if (
            input_name in self._tensor_is_spatial
            and self._tensor_is_spatial[input_name]
        ):
            # Store the shape before flattening for weight reordering
            self._tensor_is_spatial[output_name] = input_vars.shape

    def _process_dropout(self, node, **kwargs):
        """Process a Dropout node (no-op during inference)."""
        input_name = node.input[0]
        input_vars = self.get_tensor_vars(input_name)

        # Dropout is a no-op during inference
        output_name = node.output[0]
        self.set_tensor_vars(output_name, input_vars)

    def _process_batchnorm(self, node, **kwargs):
        """Process a BatchNormalization node.

        BatchNormalization: Y = gamma * (X - mean) / sqrt(variance + epsilon) + beta
        Inputs: X, gamma (scale), beta (bias), mean, variance
        """
        input_name = node.input[0]
        input_vars = self.get_tensor_vars(input_name)

        # Verify we have all required inputs
        if len(node.input) < 5:
            raise NoModel(
                self._onnx_graph,
                "BatchNormalization node requires 5 inputs: X, scale, bias, mean, var",
            )

        # Get the parameters from initializers
        gamma_name = node.input[1]
        beta_name = node.input[2]
        mean_name = node.input[3]
        var_name = node.input[4]

        if gamma_name not in self._init_map:
            raise NoModel(
                self._onnx_graph, "BatchNormalization gamma must be an initializer"
            )
        if beta_name not in self._init_map:
            raise NoModel(
                self._onnx_graph, "BatchNormalization beta must be an initializer"
            )
        if mean_name not in self._init_map:
            raise NoModel(
                self._onnx_graph, "BatchNormalization mean must be an initializer"
            )
        if var_name not in self._init_map:
            raise NoModel(
                self._onnx_graph, "BatchNormalization variance must be an initializer"
            )

        gamma = self._init_map[gamma_name].reshape(-1)
        beta = self._init_map[beta_name].reshape(-1)
        mean = self._init_map[mean_name].reshape(-1)
        variance = self._init_map[var_name].reshape(-1)

        # Get epsilon attribute (default 1e-5 in ONNX)
        epsilon = self._get_attr(node, "epsilon", 1e-5)

        # Add batch normalization layer
        layer = self._add_batchnorm_layer(
            input_vars,
            gamma,
            beta,
            mean,
            variance,
            epsilon,
            name=node.name,
            **kwargs,
        )

        output_name = node.output[0]
        self.set_tensor_vars(output_name, layer.output)

        # Preserve spatial marking if input was spatial
        if input_name in self._tensor_is_spatial:
            self._tensor_is_spatial[output_name] = self._tensor_is_spatial[input_name]

    def get_error(self, eps=None):
        """Compute prediction error against ONNX Runtime."""
        if self._has_solution:
            import onnxruntime as ort

            sess = ort.InferenceSession(self.predictor.SerializeToString())
            input_name = sess.get_inputs()[0].name

            # Check if model has spatial layers (conv or maxpool)
            # We need to check the graph nodes for Conv or MaxPool operations
            has_spatial = any(
                node.op_type in ("Conv", "MaxPool") for node in self._onnx_graph.node
            )

            # If model has spatial layers, input_values are in NHWC format
            # but ONNX expects NCHW, so we need to convert
            input_data = self.input_values.astype(np.float32)
            if has_spatial and input_data.ndim == 4:
                # Convert from NHWC to NCHW for ONNX inference
                input_data = np.transpose(input_data, (0, 3, 1, 2))

            pred = sess.run(None, {input_name: input_data})[0]

            # If output is 4D and model has spatial layers, convert back from NCHW to NHWC
            if has_spatial and pred.ndim == 4:
                pred = np.transpose(pred, (0, 2, 3, 1))

            r_val = np.abs(pred - self.output_values)
            if eps is not None and np.max(r_val) > eps:
                print(f"{pred} != {self.output_values}")
            return r_val
        raise NoSolution()
