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

"""ONNX support for formulating sequential neural networks.

Supports neural networks represented with:
- Dense layers: ONNX `Gemm` nodes or `MatMul`+`Add` sequences
- Convolutional layers: `Conv` nodes (2D, valid padding only)
- Pooling: `MaxPool` nodes (valid padding only)
- `Flatten` nodes
- `Relu` activations

This mirrors the capabilities of the Keras and PyTorch converters.

Note: For models with convolutional layers, input variables should be provided
in NHWC format (batch, height, width, channels), even though ONNX models use
NCHW format internally.
"""

from .onnx_model import add_onnx_constr as add_onnx_constr  # noqa: F401
