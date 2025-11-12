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
"""
VNN-LIB Support for Gurobi Machine Learning.

This subpackage provides tools for working with VNN-LIB (Verification of Neural Networks Library)
format files commonly used in neural network verification competitions.

Main functionality:
- Parsing VNN-LIB property specification files
- Converting ONNX + VNN-LIB to Gurobi MPS optimization models
- Support for input bounds and output constraints (both conjunctive and disjunctive)

Example usage:
    >>> from gurobi_ml.vnnlib import parse_vnnlib_simple, convert_to_mps
    >>>
    >>> # Parse VNN-LIB property file
    >>> prop = parse_vnnlib_simple('property.vnnlib')
    >>> print(f"Inputs: {prop.num_inputs}, Outputs: {prop.num_outputs}")
    >>>
    >>> # Convert ONNX + VNN-LIB to MPS
    >>> success, msg = convert_to_mps('model.onnx', 'property.vnnlib', 'output.mps')

Note:
    VNN-LIB support is primarily intended for verification benchmarks and competitions.
    For general neural network integration with Gurobi, use the main gurobi_ml.onnx module.
"""

from .parser import parse_vnnlib_simple, parse_vnnlib, VNNLIBProperty
from .converter import convert_to_mps

__all__ = [
    "parse_vnnlib_simple",
    "parse_vnnlib",
    "VNNLIBProperty",
    "convert_to_mps",
]
