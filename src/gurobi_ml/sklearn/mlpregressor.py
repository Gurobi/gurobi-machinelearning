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

""" Module for embeding a :external+sklearn:py:class:`sklearn.neural_network.MLPRegressor` into a
:gurobipy:`model`
"""
from ..exceptions import NoModel
from ..modeling.neuralnet import BaseNNConstr
from .skgetter import SKgetter


def add_mlp_regressor_constr(gp_model, mlp_regressor, input_vars, output_vars=None, **kwargs):
    """Embed mlp_regressor into gp_model

    Predict the values of output_vars using input_vars

    Parameters
    ----------
    gp_model: :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    mlpregressor: :external+sklearn:py:class:`sklearn.neural_network.MLPRegressor`
        The multi-layer perceptron regressor to insert as predictor.
    input_vars: :gurobipy:`mvar` or :gurobipy:`var` array like
        Decision variables used as input for regression in model.
    output_vars: :gurobipy:`mvar` or :gurobipy:`var` array like, optional
        Decision variables used as output for regression in model.

    Returns
    -------
    MLPRegressorConstr
        Object containing information about what was added to gp_model to embed the
        predictor into it

    Raises
    ------
    NoModel
        If the translation to Gurobi of the activation function for the network
        is not implemented.

    Note
    ----
    |VariablesDimensionsWarn|
    """
    return MLPRegressorConstr(gp_model, mlp_regressor, input_vars, output_vars, **kwargs)


class MLPRegressorConstr(SKgetter, BaseNNConstr):
    """Predict a Gurobi matrix variable using a neural network that
    takes another Gurobi matrix variable as input.
    """

    def __init__(
        self,
        gp_model,
        predictor,
        input_vars,
        output_vars=None,
        clean_predictor=False,
        **kwargs,
    ):
        SKgetter.__init__(self, predictor, **kwargs)
        BaseNNConstr.__init__(
            self,
            gp_model,
            predictor,
            input_vars,
            output_vars,
            clean_predictor=clean_predictor,
            **kwargs,
        )
        assert predictor.out_activation_ in ("identity", "relu")

    def _mip_model(self, **kwargs):
        """Add the prediction constraints to Gurobi"""
        neural_net = self.predictor
        if neural_net.activation not in self.act_dict:
            print(self.act_dict)
            raise NoModel(
                neural_net,
                f"No implementation for activation function {neural_net.activation}",
            )
        activation = self.act_dict[neural_net.activation]

        input_vars = self._input
        output = None

        for i in range(neural_net.n_layers_ - 1):
            layer_coefs = neural_net.coefs_[i]
            layer_intercept = neural_net.intercepts_[i]

            # For last layer change activation
            if i == neural_net.n_layers_ - 2:
                activation = self.act_dict[neural_net.out_activation_]
                output = self._output

            layer = self.add_dense_layer(
                input_vars,
                layer_coefs,
                layer_intercept,
                activation,
                output,
            )
            input_vars = layer._output  # pylint: disable=W0212
            self._gp_model.update()
        assert (
            self._output is not None
        )  # Should never happen since sklearn object defines n_ouputs_
