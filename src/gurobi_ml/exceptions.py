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


class NotRegistered(Exception):
    def __init__(self, predictor):
        super().__init__("Object of type {} is not registered/supported with gurobi_ml".format(type(predictor).__name__))


class NoModel(Exception):
    def __init__(self, predictor, reason):
        super().__init__("Can't do model for {}: {}".format(type(predictor).__name__, reason))


class NoSolution(Exception):
    def __init__(self):
        super().__init__("No solution available")


class ModelingError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ParameterError(Exception):
    def __init__(self, message):
        super().__init__(message)


class InternalError(Exception):
    def __init__(self, message):
        super().__init__(message)
