# Copyright © 2023-2026 Gurobi Optimization, LLC
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

"""Bases classes for modeling neural network layers."""

from .._var_utils import _default_name
from ..base_predictor_constr import AbstractPredictorConstr
from .activations import Identity, ReLU, SqrtReLU, SoftReLU
from .layers import ActivationLayer, DenseLayer


class BaseNNConstr(AbstractPredictorConstr):
    """Base class for inserting a regressor based on a neural-network/tensor into Gurobi.

    This only supports sequential neural networks.

    The bracket operator can be used to iterate over the layers of the layers of the network.
    It will give access to the modeling object of the corresponding layer.

    """

    def __init__(self, gp_model, predictor, input_vars, output_vars, **kwargs):
        self.predictor = predictor

        # Initialize default activations
        # Note: softplus is initialized lazily to avoid breaking older Gurobi versions
        self.act_dict = {
            "relu": ReLU(),
            "identity": Identity(),
        }

        # Support convenient relu_formulation parameter to replace ReLU
        relu_formulation = kwargs.get("relu_formulation", "mip")
        if relu_formulation == "smooth":
            self.act_dict["relu"] = SqrtReLU()
        elif relu_formulation == "soft":
            beta = kwargs.get("soft_relu_beta", 1.0)
            self.act_dict["relu"] = SoftReLU(beta=beta)
        elif relu_formulation != "mip":
            raise ValueError(
                f"relu_formulation must be 'mip', 'smooth', or 'soft', got '{relu_formulation}'"
            )

        # Allow custom activation_models to override defaults
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

    def _get_activation(self, activation_name):
        """Get activation model, initializing softplus lazily if needed.
        
        Parameters
        ----------
        activation_name : str
            Name of the activation function
            
        Returns
        -------
        Activation model object
        """
        if activation_name not in self.act_dict and activation_name == "softplus":
            # Lazy initialization for softplus to support older Gurobi versions
            self.act_dict["softplus"] = SoftReLU(beta=1.0)
        return self.act_dict[activation_name]

    def __iter__(self):
        """Iterate over layers of neural network"""
        return self.layers.__iter__()

    @property
    def layers(self):
        """Access models for successive layers of the network"""
        return self._layers

    def _add_dense_layer(
        self,
        input_vars,
        layer_coefs,
        layer_intercept,
        activation,
        activation_vars=None,
        **kwargs,
    ):
        """Add a dense layer to gurobipy model.

        Parameters
        ----------

        input_vars : mvar_array_like
            Decision variables used as input for predictor in model.
        layer_coefs:
            Coefficient for each node in a layer
        layer_intercept:
            Intercept bias
        activation:
            Activation function
        activation_vars : None, optional
            Output variables
        """
        layer = DenseLayer(
            self.gp_model,
            activation_vars,
            input_vars,
            layer_coefs,
            layer_intercept,
            activation,
            **kwargs,
        )
        self._layers.append(layer)
        return layer

    def _add_activation_layer(
        self, input_vars, activation, activation_vars=None, **kwargs
    ):
        """Add an activation layer to gurobipy model.

        Parameters
        ----------

        input_vars : mvar_array_like
            Decision variables used as input for predictor in gurobipy model.
        activation:
            Activation function
        activation_vars : mvar_array_like, optional
            Output variables
        """
        layer = ActivationLayer(
            self.gp_model, activation_vars, input_vars, activation, **kwargs
        )
        self._layers.append(layer)
        return layer

    def print_stats(self, abbrev=False, file=None):
        """Print statistics about submodel created.

        Parameters
        ----------

        file : None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """
        if not abbrev:
            super().print_stats(abbrev, file)
            print(file=file)

            self._print_container_steps("Layer", self._layers, file=file)
        else:
            for layer in self:
                layer.print_stats(abbrev=True, file=file)
                print(file=file)
