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

""" Define generic function that can add any known trained predictor
"""
import sys

from .register_predictor import user_predictors


def sklearn_convertors():
    if "sklearn" in sys.modules:
        from .sklearn import add_pipeline_constr
        from .sklearn.predictors_list import sklearn_predictors, sklearn_transformers

        return (
            sklearn_predictors()
            | sklearn_transformers()
            | {
                "Pipeline": add_pipeline_constr,
            }
        )
    else:
        return {}


def pytorch_convertors():
    """Collect known PyTorch objects that can be embedded and the conversion class"""
    if "torch" in sys.modules:
        import torch  # pylint: disable=import-outside-toplevel

        from .torch import (
            add_sequential_constr as add_torch_sequential_constr,  # pylint: disable=import-outside-toplevel
        )

        return {torch.nn.Sequential: add_torch_sequential_constr}
    return {}


def keras_convertors():
    """Collect known Keras objects that can be embedded and the conversion class"""
    if "tensorflow" in sys.modules:
        from keras.engine.functional import (
            Functional,  # pylint: disable=import-outside-toplevel
        )
        from keras.engine.training import (
            Model,  # pylint: disable=import-outside-toplevel
        )
        from tensorflow import keras  # pylint: disable=import-outside-toplevel

        from .keras import (
            add_keras_constr as add_keras_constr,  # pylint: disable=import-outside-toplevel
        )

        return {
            keras.Sequential: add_keras_constr,
            Functional: add_keras_constr,
            Model: add_keras_constr,
        }
    return {}


def registered_predictors():
    """Return the list of registered predictors"""
    convertors = {}
    convertors |= sklearn_convertors()
    convertors |= pytorch_convertors()
    convertors |= keras_convertors()
    convertors |= user_predictors()
    return convertors
