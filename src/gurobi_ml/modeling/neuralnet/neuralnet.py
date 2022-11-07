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

"""Bases classes for modeling neural network layers"""


from ..basepredictor import AbstractPredictorConstr, _default_name
from .activations import Identity, ReLU
from .layers import ActivationLayer, DenseLayer


class BaseNNConstr(AbstractPredictorConstr):
    """Base class for inserting a regressor based on neural-network/tensor into Gurobi"""

    def __init__(self, gp_model, predictor, input_vars, output_vars, **kwargs):
        self.predictor = predictor
        self.act_dict = {
            "relu": ReLU(),
            "identity": Identity(),
        }
        try:
            for activation, activation_model in kwargs["activation_models"].items():
                self.act_dict[activation] = activation_model
        except KeyError:
            pass
        self._layers = []

        self._default_name = _default_name(predictor)
        super().__init__(
            gp_model=gp_model,
            input_vars=input_vars,
            output_vars=output_vars,
            **kwargs,
        )

    def __iter__(self):
        return self._layers.__iter__()

    def add_dense_layer(
        self,
        input_vars,
        layer_coefs,
        layer_intercept,
        activation,
        activation_vars=None,
        **kwargs,
    ):
        """Add a dense layer to gurobipy model

        Parameters
        ---------

        input_vars:  mvar_array_like
            Decision variables used as input for predictor in model.
        layer_coefs:
            Coefficient for each node in a layer
        layer_intercept:
            Intercept bias
        activation:
            Activation function
        activation_vars: None, optional
            Output variables
        """
        layer = DenseLayer(
            self._gp_model,
            activation_vars,
            input_vars,
            layer_coefs,
            layer_intercept,
            activation,
            **kwargs,
        )
        self._layers.append(layer)
        return layer

    def add_activation_layer(self, input_vars, activation, activation_vars=None, **kwargs):
        """Add an activation layer to gurobipy model

        Parameters
        ---------

        input_vars:  mvar_array_like
            Decision variables used as input for predictor in gurobipy model.
        activation:
            Activation function
        activation_vars: mvar_array_like, optional
            Output variables
        """
        layer = ActivationLayer(self._gp_model, activation_vars, input_vars, activation, **kwargs)
        self._layers.append(layer)
        return layer

    def print_stats(self, abbrev=False, file=None):
        """Print statistics about submodel created

        Parameters
        ---------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """
        super().print_stats(abbrev, file)
        if abbrev:
            return
        print(file=file)

        header = f"{'Layer':12} {'Activation':12} {'Output Shape':>12} {'Variables':>10} {'Constraints':^38}"
        print("-" * len(header), file=file)
        print(header, file=file)
        print(f"{' '*50} {'Linear':>10} {'Quadratic':>10} {'General':>10}", file=file)
        print("=" * len(header), file=file)
        for layer in self:
            layer.print_stats(file=file)
            print(file=file)
        print("-" * len(header), file=file)
