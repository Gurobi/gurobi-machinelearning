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

"""Define generic function that can add any known trained predictor."""
import sys

from .register_user_predictor import user_predictors


def sklearn_convertors():
    """Collect convertors for Scikit-learn objects."""
    if "sklearn" in sys.modules:
        from .sklearn import (  # pylint: disable=import-outside-toplevel
            add_pipeline_constr,
        )
        from .sklearn.predictors_list import (  # pylint: disable=import-outside-toplevel
            sklearn_predictors,
        )
        from .sklearn.preprocessing import (
            sklearn_transformers,  # pylint: disable-import-outside-toplevel
        )

        return (
            sklearn_predictors()
            | sklearn_transformers()
            | {
                "Pipeline": add_pipeline_constr,
            }
        )
    return {}


def pytorch_convertors():
    """Collect known PyTorch objects that can be formulated and the conversion class."""
    if "torch" in sys.modules:
        import torch  # pylint: disable=import-outside-toplevel

        from .torch import (  # pylint: disable=import-outside-toplevel
            add_sequential_constr,
        )

        return {torch.nn.Sequential: add_sequential_constr}
    return {}


def xgboost_convertors():
    """Collect known PyTorch objects that can be formulated and the conversion class."""
    if "xgboost" in sys.modules:
        import xgboost as xgb  # pylint: disable=import-outside-toplevel

        from .xgboost import (  # pylint: disable=import-outside-toplevel
            add_xgboost_regressor_constr,
            add_xgbregressor_constr,
        )

        return {
            xgb.core.Booster: add_xgboost_regressor_constr,
            xgb.XGBRegressor: add_xgbregressor_constr,
        }
    return {}


def keras_convertors():
    """Collect known Keras objects that can be embedded and the conversion class."""
    if "tensorflow" in sys.modules:
        from tensorflow import keras  # pylint: disable=import-outside-toplevel

        from .keras import add_keras_constr  # pylint: disable=import-outside-toplevel

        return {
            keras.Sequential: add_keras_constr,
            keras.Model: add_keras_constr,
        }
    return {}


def registered_predictors():
    """Return the list of registered predictors."""
    convertors = {}
    convertors |= sklearn_convertors()
    convertors |= pytorch_convertors()
    convertors |= keras_convertors()
    convertors |= xgboost_convertors()
    convertors |= user_predictors()
    return convertors
