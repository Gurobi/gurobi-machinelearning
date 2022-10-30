# Copyright © 2022 Gurobi Optimization, LLC
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

""" Exceptions for gurobi_ml """


class NotRegistered(Exception):
    """Predictor is not supported by gurobi_ml"""

    def __init__(self, predictor):
        super().__init__(f"Object of type {predictor} is not registered/supported with gurobi_ml")


class NoModel(Exception):
    """No model is known for some structure"""

    def __init__(self, predictor, reason):
        if not isinstance(predictor, str):
            predictor = type(predictor).__name__
        super().__init__(f"Can't do model for {predictor}: {reason}")


class NoSolution(Exception):
    """Gurobi doesn't have a solution"""

    def __init__(self):
        super().__init__("No solution available")


class ParameterError(Exception):
    """Wrong parameter to a function"""

    def __init__(self, message):
        super().__init__(message)
