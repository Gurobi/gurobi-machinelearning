# Copyright © 2023-2025 Gurobi Optimization, LLC
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

"""Convertor for LigthGBM's Scikit Learn compatible object

This is to be able to use it in pipelines.
"""

import sys


def lightgbm_sklearn_convertors():
    """Collect known PyTorch objects that can be formulated and the conversion class."""
    if "lightgbm" in sys.modules:
        import lightgbm as lgbm  # pylint: disable=import-outside-toplevel

        from .lightgbm import add_lgbmregressor_constr

        return {
            lgbm.sklearn.LGBMRegressor: add_lgbmregressor_constr,
        }
    return {}
