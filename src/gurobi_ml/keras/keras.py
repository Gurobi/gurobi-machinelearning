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

"""Module for formulating a Keras model into a :external+gurobi:py:class:`Model`."""

import numpy as np
import keras

from ..exceptions import NoModel, NoSolution
from ..modeling.neuralnet import BaseNNConstr


def add_keras_constr(gp_model, keras_model, input_vars, output_vars=None, **kwargs):
    """Formulate keras_model into gp_model.

    The formulation predicts the values of output_vars using input_vars according to keras_model.
    See our :ref:`Users Guide <Neural Networks>` for details on the mip formulation used.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    keras_model : `keras.Model <https://keras.io/api/models/model/>`
        The keras model to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for Keras model in gp_model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for Keras model in gp_model.

    Returns
    -------
    KerasNetworkConstr
        Object containing information about what was added to gp_model to formulate
        keras_model into it

    Raises
    ------
    NoModel
        If the translation for some of the Keras model structure
        (layer or activation) is not implemented.

    Warnings
    --------

      Only `Dense <https://keras.io/api/layers/core_layers/dense/>`_ (possibly
      with `relu` activation), and `ReLU <https://keras.io/api/layers/activation_layers/relu/>`_ with
      default settings are supported.

    Notes
    -----
    |VariablesDimensionsWarn|

    **Models with Skip Connections or Residual Architectures**

    This function only supports sequential Keras models (linear chains of layers).
    For models with more complex architectures such as:

    - Skip connections (input reused by multiple layers)
    - Residual connections (intermediate outputs reused)
    - Multi-branch architectures
    - Keras Functional API models with non-sequential topology

    **Use the ONNX export workflow instead:**

    1. Export your Keras model to ONNX format:

       .. code-block:: python

           import tf2onnx
           import onnx

           # Method 1: Using tf2onnx (recommended)
           spec = (tf.TensorSpec((None, input_dim), tf.float32, name="input"),)
           model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec)
           onnx.save(model_proto, "model.onnx")

           # Method 2: Using keras2onnx (alternative)
           import keras2onnx
           onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
           onnx.save_model(onnx_model, "model.onnx")

    2. Load and use with Gurobi ML's ONNX DAG support:

       .. code-block:: python

           import onnx
           from gurobi_ml.onnx import add_onnx_dag_constr

           onnx_model = onnx.load("model.onnx")
           pred = add_onnx_dag_constr(gp_model, onnx_model, input_vars)

    For more details on exporting Keras models to ONNX, see:

    - `tf2onnx documentation <https://github.com/onnx/tensorflow-onnx>`_
    - `keras2onnx documentation <https://github.com/onnx/keras-onnx>`_
    - `ONNX tutorials <https://onnx.ai/get-started.html>`_
    """
    return KerasNetworkConstr(gp_model, keras_model, input_vars, output_vars, **kwargs)


class KerasNetworkConstr(BaseNNConstr):
    """Class to formulate a trained `keras.Model <https://keras.io/api/models/model/>` in a gurobipy model.

    |ClassShort|
    """

    def __init__(self, gp_model, predictor, input_vars, output_vars=None, **kwargs):
        assert predictor.built
        for step in predictor.layers:
            if isinstance(step, (keras.layers.Dense)):
                config = step.get_config()
                activation = config["activation"]
                if activation == "softmax":
                    pass
                elif activation not in ("relu", "linear"):
                    raise NoModel(predictor, f"Unsupported activation {activation}")
            elif isinstance(step, (keras.layers.Conv2D)):
                config = step.get_config()
                activation = config["activation"]
                if activation == "softmax":
                    pass
                elif activation not in ("relu", "linear"):
                    raise NoModel(predictor, f"Unsupported activation {activation}")
                kwargs["accepted_dim"] = (4,)
            elif isinstance(
                step,
                (keras.layers.MaxPooling2D, keras.layers.Flatten, keras.layers.Dropout),
            ):
                pass
            elif isinstance(step, keras.layers.ReLU):
                if step.negative_slope != 0.0:
                    raise NoModel(
                        predictor, "Only handle ReLU layers with negative slope 0.0"
                    )
                if step.threshold != 0.0:
                    raise NoModel(
                        predictor, "Only handle ReLU layers with threshold of 0.0"
                    )
                if step.max_value is not None and step.max_value < float("inf"):
                    raise NoModel(predictor, "Only handle ReLU layers without maxvalue")
            elif isinstance(step, keras.layers.InputLayer):
                pass
            else:
                raise NoModel(
                    predictor, f"Unsupported network layer {type(step).__name__}"
                )

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
                layer = self._add_activation_layer(
                    _input, self.act_dict["relu"], output, name=f"relu{i}", **kwargs
                )
                _input = layer.output
            elif isinstance(step, keras.layers.Dense):
                config = step.get_config()
                activation = config["activation"]
                if activation == "linear":
                    activation = "identity"
                weights, bias = step.get_weights()
                kwargs["accepted_dim"] = (2,)
                layer = self._add_dense_layer(
                    _input,
                    weights,
                    bias,
                    self.act_dict[activation],
                    output,
                    name=f"dense{i}",
                    **kwargs,
                )
                _input = layer.output
            elif isinstance(step, keras.layers.Conv2D):
                config = step.get_config()
                activation = config["activation"]
                if activation == "linear":
                    activation = "identity"
                weights, bias = step.get_weights()
                kwargs["accepted_dim"] = (4,)
                layer = self._add_conv2d_layer(
                    _input,
                    weights,
                    bias,
                    config["filters"],
                    config["kernel_size"],
                    config["strides"],
                    config["padding"],
                    self.act_dict[activation],
                    output,
                    name=f"conv2d{i}",
                    **kwargs,
                )
                _input = layer.output
            elif isinstance(step, keras.layers.MaxPooling2D):
                config = step.get_config()
                kwargs["accepted_dim"] = (4,)
                layer = self._add_maxpool2d_layer(
                    _input,
                    config["pool_size"],
                    config["strides"],
                    config["padding"],
                    output,
                    name=f"maxpool2d{i}",
                    **kwargs,
                )
                _input = layer.output
            elif isinstance(step, keras.layers.Flatten):
                kwargs["accepted_dim"] = (2,)
                layer = self._add_flatten_layer(
                    _input,
                    output,
                    name=f"flatten{i}",
                    **kwargs,
                )
                _input = layer.output
            elif isinstance(step, keras.layers.Dropout):
                layer = self._add_activation_layer(
                    _input,
                    self.act_dict["identity"],
                    output,
                    name=f"dropout{i}",
                    **kwargs,
                )
                _input = layer.output
        if self._output is None:
            self._output = layer.output

    def get_error(self, eps=None):
        if self._has_solution:
            r_val = np.abs(
                self.predictor.predict(self.input_values) - self.output_values
            )
            if eps is not None and np.max(r_val) > eps:
                print(f"{self.input_values} != {self.output_values}")
            return r_val
        raise NoSolution()
