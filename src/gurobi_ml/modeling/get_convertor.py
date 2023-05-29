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

"""Utility function to find function that add a predictor in dictionnary."""


def get_convertor(predictor, convertors):
    """Return the convertor for a given predictor."""
    convertor = None
    try:
        convertor = convertors[type(predictor)]
    except KeyError:
        pass
    if convertor is None:
        for parent in type(predictor).mro():
            try:
                convertor = convertors[parent]
                break
            except KeyError:
                pass
    if convertor is None:
        name = type(predictor).__name__
        try:
            convertor = convertors[name]
        except KeyError:
            pass
    return convertor
