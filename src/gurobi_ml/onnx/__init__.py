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

"""ONNX support for formulating neural networks.

Supports neural networks represented with:
- Dense layers: ONNX `Gemm` nodes or `MatMul`+`Add` sequences
- Convolutional layers: `Conv` nodes (2D, with symmetric padding)
- Pooling: `MaxPool` nodes (2D, with symmetric padding)
- `Flatten` nodes
- `Relu` activations

This mirrors the capabilities of the Keras and PyTorch converters.

Padding support: Conv and MaxPool support symmetric padding where pad_left = pad_right
and pad_top = pad_bottom. Asymmetric padding is not supported.

Note: For models with convolutional layers, input variables should be provided
in NHWC format (batch, height, width, channels), even though ONNX models use
NCHW format internally.

Provides two implementations:

1. **add_onnx_constr**: Sequential feed-forward networks only

   - Validates that models have sequential topology
   - Rejects models with skip/residual connections
   - Use for simple feed-forward networks

2. **add_onnx_dag_constr**: Supports arbitrary DAG topologies

   - Skip connections (input used by multiple layers)
   - Residual connections (intermediate outputs reused)
   - Multi-branch architectures
   - Any valid directed acyclic graph

**Recommended Workflow for Keras and PyTorch Models:**

For models with complex architectures (ResNet, skip connections, etc.),
export to ONNX first and use add_onnx_dag_constr:

From Keras:
    >>> import tf2onnx
    >>> import onnx
    >>> spec = (tf.TensorSpec((None, input_dim), tf.float32, name="input"),)
    >>> model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec)
    >>> onnx.save(model_proto, "model.onnx")

From PyTorch:
    >>> import torch
    >>> dummy_input = torch.randn(1, input_dim)
    >>> torch.onnx.export(pytorch_model, dummy_input, "model.onnx")

Then use with Gurobi ML:
    >>> onnx_model = onnx.load("model.onnx")
    >>> from gurobi_ml.onnx import add_onnx_dag_constr
    >>> pred = add_onnx_dag_constr(gp_model, onnx_model, input_vars)
"""

from .onnx_dag_model import add_onnx_dag_constr as add_onnx_dag_constr  # noqa: F401
from .onnx_model import add_onnx_constr as add_onnx_constr  # noqa: F401
