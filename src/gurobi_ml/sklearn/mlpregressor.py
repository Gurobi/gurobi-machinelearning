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

""" Module for inserting an :external+sklearn:py:class:`sklearn.neural_network.MLPRegressor` into a gurobipy model
"""
from ..exceptions import NoModel
from ..modeling.neuralnet import BaseNNConstr
from .skgetter import SKgetter


class MLPRegressorConstr(SKgetter, BaseNNConstr):
    """Predict a Gurobi matrix variable using a neural network that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, predictor, input_vars, output_vars=None, clean_predictor=False, **kwargs):
        SKgetter.__init__(self, predictor, **kwargs)
        BaseNNConstr.__init__(
            self,
            grbmodel,
            predictor,
            input_vars,
            output_vars,
            clean_predictor=clean_predictor,
            **kwargs,
        )
        assert predictor.out_activation_ in ("identity", "relu")

    def _mip_model(self):
        """Add the prediction constraints to Gurobi"""
        neuralnet = self.predictor
        if neuralnet.activation not in self.actdict:
            print(self.actdict)
            raise NoModel(neuralnet, f"No implementation for activation function {neuralnet.activation}")
        activation = self.actdict[neuralnet.activation]

        input_vars = self._input
        output = None

        for i in range(neuralnet.n_layers_ - 1):
            layer_coefs = neuralnet.coefs_[i]
            layer_intercept = neuralnet.intercepts_[i]

            # For last layer change activation
            if i == neuralnet.n_layers_ - 2:
                activation = self.actdict[neuralnet.out_activation_]
                output = self._output

            layer = self.add_dense_layer(input_vars, layer_coefs, layer_intercept, activation, output, name=f"layer{i}")
            input_vars = layer._output  # pylint: disable=W0212
            self._model.update()
        assert self._output is not None  # Should never happen since sklearn object defines n_ouputs_


def add_mlp_regressor_constr(grbmodel, mlpregressor, input_vars, output_vars=None, **kwargs):
    """Use a `decision_tree_regressor` to predict the value of `output_vars` using `input_vars` in `grbmodel`

    Parameters
    ----------
    grbmodel: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
        The gurobipy model where the predictor should be inserted.
    mlpregressor: :external+sklearn:py:class:`sklearn.neural_network.MLPRegressor`
        The multi-layer perceptron regressor to insert as predictor.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    MLPRegressorConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Raises
    ------
    NoModel
        If the translation to Gurobi of the activation function for the network
        is not implemented.

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars
    """
    return MLPRegressorConstr(grbmodel, mlpregressor, input_vars, output_vars, **kwargs)
