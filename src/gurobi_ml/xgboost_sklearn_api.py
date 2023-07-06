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

""" Convertor for XGBoost's Scikit Learn compatible object

This is to be able to use it in pipelines.
"""

import sys


def xgboost_sklearn_convertors():
    """Collect known PyTorch objects that can be formulated and the conversion class."""
    if "xgboost" in sys.modules:
        import xgboost as xgb  # pylint: disable=import-outside-toplevel

        from .xgboost import add_xgbregressor_constr

        return {
            xgb.XGBRegressor: add_xgbregressor_constr,
        }
    return {}
