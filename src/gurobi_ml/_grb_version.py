# Copyright © 2023-2026 Gurobi Optimization, LLC
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

"""Centralized Gurobi version capability flags."""

import gurobipy as gp

GUROBI_VERSION = gp.gurobi.version()

# Gurobi 11+: FuncNonlinear general constraint attribute and MVar-compatible indicator constraints
HAS_FUNCNONLINEAR = GUROBI_VERSION >= (11, 0, 0)

# Gurobi 12+: gurobipy.nlfunc module for smooth nonlinear activations (sigmoid, softplus, …)
HAS_NLFUNC = GUROBI_VERSION >= (12, 0, 0)

# Gurobi 13+: tanh support via gurobipy.nlfunc
HAS_TANH = GUROBI_VERSION >= (13, 0, 0)
