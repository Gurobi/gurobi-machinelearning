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

Provides two implementations:
1. add_onnx_constr: Sequential feed-forward networks only
2. add_onnx_dag_constr: Supports arbitrary DAG topologies including skip/residual connections

The DAG-based implementation supports:
- Skip connections (input used by multiple layers)
- Residual connections (intermediate outputs reused)
- Multi-branch architectures
"""

from .onnx_dag_model import add_onnx_dag_constr as add_onnx_dag_constr  # noqa: F401
from .onnx_model import add_onnx_constr as add_onnx_constr  # noqa: F401
