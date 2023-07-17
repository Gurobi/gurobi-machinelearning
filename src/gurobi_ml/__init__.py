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

"""
Gurobi Machine learning
==================================

A Python package to help use trained regression models in
mathematical optimization models. The package supports a variety of regression models
(linear, logistic, neural networks, decision trees,...) trained by
different machine learning frameworks (scikit-learn, Keras and PyTorch).

See https://gurobi-optimization-gurobi-machine-learning.readthedocs-hosted.com/
for documentation.
"""

# read version from installed package

from gurobipy import gurobi

from ._version import __version__
from .add_predictor import add_predictor_constr
from .register_user_predictor import register_predictor_constr

MIN_GRB_VERSION = 10
if gurobi.version()[0] < MIN_GRB_VERSION:
    raise ImportError(
        "Gurobi version should be at least {}.0.0".format(MIN_GRB_VERSION)
    )
