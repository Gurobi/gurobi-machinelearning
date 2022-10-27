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

from .sklearn.predictors_list import sklearn_predictors

USER_PREDICTORS = {}


def pytorch_convertors():
    """Collect known PyTorch objects that can be embedded and the conversion class"""
    if "torch" in sys.modules:
        from torch import nn as pytorchnn

        from .torch import SequentialConstr as TorchSequential

        return {pytorchnn.Sequential: TorchSequential}
    return {}


def keras_convertors():
    """Collect known Keras objects that can be embedded and the conversion class"""
    if "tensorflow" in sys.modules:
        from keras.engine.functional import Functional
        from keras.engine.training import Model
        from tensorflow import keras

        from .keras import KerasNetworkConstr as KerasPredictor

        return {keras.Sequential: KerasPredictor, Functional: KerasPredictor, Model: KerasPredictor}
    return {}


def register_predictor_constr(predictor, predictor_constr):
    """Register a new predictor that can be added using use_predictor_constr

    Parameters
    ----------
    predictor:
        Class of the predictor
    predictor_constr:
        Class implementing the MIP model that embeds a trained object of
        class predictor in a gurobi Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>
    """
    USER_PREDICTORS[predictor] = predictor_constr


def registered_predictors():
    """Return the list of registered predictors"""
    convertors = {}
    convertors |= sklearn_predictors()
    convertors |= pytorch_convertors()
    convertors |= keras_convertors()
    convertors |= USER_PREDICTORS
    return convertors
