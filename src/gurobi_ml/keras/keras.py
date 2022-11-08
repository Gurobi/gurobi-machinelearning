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

""" Module for embedding a Keras model into a :gurobipy:`model`
"""

import numpy as np
from tensorflow import keras

from ..exceptions import NoModel, NoSolution
from ..modeling.neuralnet import BaseNNConstr


def add_keras_constr(gp_model, keras_model, input_vars, output_vars=None, **kwargs):
    """Embed keras_model into gp_model

    Predict the values of output_vars using input_vars

    Parameters
    ----------
    gp_model: :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    keras_model: `keras.Model <https://keras.io/api/models/model/>`
        The keras model to insert as predictor.
    input_vars: :gurobipy:`mvar` or :gurobipy:`var` array like
        Decision variables used as input for Keras model in gp_model.
    output_vars: :gurobipy:`mvar` or :gurobipy:`var` array like, optional
        Decision variables used as output for Keras model in gp_model.

    Returns
    -------
    KerasNetworkConstr
        Object containing information about what was added to gp_model to embed the
        predictor into it


    Warning
    -------

      Only `Dense <https://keras.io/api/layers/core_layers/dense/>`_ (possibly
      with `relu` activation), and `ReLU <https://keras.io/api/layers/activation_layers/relu/>`_ with
      default settings are supported.

    Raises
    ------
    NoModel
        If the translation for some of the Keras model structure
        (layer or activation) is not implemented.

    Note
    ----
    |VariablesDimensionsWarn|
    """
    return KerasNetworkConstr(gp_model, keras_model, input_vars, output_vars, **kwargs)


class KerasNetworkConstr(BaseNNConstr):
    def __init__(self, gp_model, predictor, input_vars, output_vars=None, **kwargs):
        assert predictor.built
        for step in predictor.layers:
            if isinstance(step, keras.layers.Dense):
                config = step.get_config()
                activation = config["activation"]
                if activation not in ("relu", "linear"):
                    raise NoModel(predictor, f"Unsupported activation {activation}")
            elif isinstance(step, keras.layers.ReLU):
                if step.negative_slope != 0.0:
                    raise NoModel(predictor, "Only handle ReLU layers with negative slope 0.0")
                if step.threshold != 0.0:
                    raise NoModel(predictor, "Only handle ReLU layers with threshold of 0.0")
                if step.max_value is not None and step.max_value < float("inf"):
                    raise NoModel(predictor, "Only handle ReLU layers without maxvalue")
            elif isinstance(step, keras.layers.InputLayer):
                pass
            else:
                raise NoModel(predictor, f"Unsupported network layer {type(step).__name__}")

        super().__init__(gp_model, predictor, input_vars, output_vars, **kwargs)

    def _mip_model(self, **kwargs):
        network = self.predictor
        _input = self._input
        output = None
        num_layers = len(network.layers)

        for i, step in enumerate(network.layers):
            if i == num_layers - 1:
                output = self._output
            if isinstance(step, keras.layers.InputLayer):
                pass
            elif isinstance(step, keras.layers.ReLU):
                layer = self.add_activation_layer(
                    _input, self.act_dict["relu"], output, name="relu"
                )
                _input = layer.output
            else:
                config = step.get_config()
                activation = config["activation"]
                if activation == "linear":
                    activation = "identity"
                weights, bias = step.get_weights()
                layer = self.add_dense_layer(
                    _input, weights, bias, self.act_dict[activation], output, name="dense"
                )
                _input = layer.output
        if self._output is None:
            self._output = layer.output

    def get_error(self):
        """Returns error in Gurobi's solution with respect to prediction from input

        Returns
        -------
        error: ndarray of same shape as :py:attr:`gurobi_ml.modeling.basepredictor.AbstractPredictorConstr.output`
            Assuming that we have a solution for the input and output variables
            `x, y`. Returns the absolute value of the differences between `predictor.forward(x)` and
            `y`. Where predictor is the Keras model this object is modeling.

        Raises
        ------
        NoSolution
            If the gurobipy model has no solution (either was not optimized or is infeasible).
        """
        if self._has_solution():
            return np.abs(self.predictor.predict(self.input.X) - self.output.X)
        raise NoSolution()
