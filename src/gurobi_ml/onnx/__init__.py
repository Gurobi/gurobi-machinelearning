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

"""ONNX support for formulating simple feed-forward neural networks.

Currently supports sequential MLPs represented with ONNX `Gemm` layers and
`Relu` activations, matching the capabilities of the Keras and PyTorch
converters (Dense/Linear + ReLU)."""

from .onnx_model import add_onnx_constr as add_onnx_constr  # noqa: F401
