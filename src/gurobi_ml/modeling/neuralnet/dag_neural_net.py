# Copyright © 2025 Gurobi Optimization, LLC
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

"""DAG (Directed Acyclic Graph) based neural network support.

This module provides support for neural networks with arbitrary graph topologies,
including skip connections, residual connections, and other non-sequential architectures.
"""

import gurobipy as gp
import numpy as np

from .._var_utils import _default_name
from ..base_predictor_constr import AbstractPredictorConstr
from .activations import Identity, ReLU
from .layers import ActivationLayer, DenseLayer


class AddLayer(AbstractPredictorConstr):
    """Layer that adds two or more input tensors element-wise."""

    def __init__(self, gp_model, output_vars, input_vars_list, **kwargs):
        """
        Parameters
        ----------
        gp_model : gurobipy.Model
            The Gurobi model
        output_vars : mvar or None
            Output variables for the addition
        input_vars_list : list of mvar
            List of input variable arrays to add together
        """
        self.input_vars_list = input_vars_list
        self._default_name = "add"
        
        # For AbstractPredictorConstr, we need a single input_vars
        # We'll use the first one as the primary input
        super().__init__(
            gp_model=gp_model,
            input_vars=input_vars_list[0],
            output_vars=output_vars,
            **kwargs,
        )

    def _create_output_vars(self, input_vars):
        """Create output variables with same shape as inputs."""
        return self.gp_model.addMVar(
            input_vars.shape, lb=-gp.GRB.INFINITY, name="add_out"
        )

    def _mip_model(self, **kwargs):
        """Add constraints: output = sum of all inputs."""
        # Create the sum of all inputs
        result = self.input_vars_list[0]
        for i in range(1, len(self.input_vars_list)):
            result = result + self.input_vars_list[i]
        
        # Constrain output to equal the sum
        self.gp_model.addConstr(self._output == result, name=f"{self._default_name}_sum")
        self.gp_model.update()

    def get_error(self, eps=None):
        """Cannot compute error for intermediate layer."""
        assert False, "Cannot compute error for AddLayer"


class DAGNNConstr(AbstractPredictorConstr):
    """Neural network with DAG topology support.

    This class supports neural networks with arbitrary directed acyclic graph
    topologies, including:
    - Skip connections (input used by multiple layers)
    - Residual connections (intermediate outputs used by multiple layers)
    - Addition nodes that combine multiple branches

    The network is represented as a computation graph where each node is either:
    - A dense (fully connected) layer
    - An activation layer
    - An addition layer (for combining branches)

    Nodes are processed in topological order to ensure dependencies are satisfied.
    """

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

        # Dictionary mapping tensor name to Gurobi variables
        self._tensor_vars = {}
        # List of layers in topological order
        self._layers = []
        # Dictionary mapping layer name to layer object
        self._layer_dict = {}

        self._default_name = _default_name(predictor)
        super().__init__(
            gp_model=gp_model,
            input_vars=input_vars,
            output_vars=output_vars,
            **kwargs,
        )

    def __iter__(self):
        """Iterate over layers in topological order."""
        return self._layers.__iter__()

    @property
    def layers(self):
        """Access models for layers of the network in topological order."""
        return self._layers

    def get_tensor_vars(self, tensor_name):
        """Get Gurobi variables for a given tensor name.

        Parameters
        ----------
        tensor_name : str
            Name of the tensor in the computation graph

        Returns
        -------
        mvar
            Gurobi variables representing the tensor
        """
        return self._tensor_vars.get(tensor_name)

    def set_tensor_vars(self, tensor_name, vars):
        """Set Gurobi variables for a given tensor name.

        Parameters
        ----------
        tensor_name : str
            Name of the tensor in the computation graph
        vars : mvar
            Gurobi variables to associate with the tensor
        """
        self._tensor_vars[tensor_name] = vars

    def _add_dense_layer(
        self,
        input_vars,
        layer_coefs,
        layer_intercept,
        activation,
        activation_vars=None,
        **kwargs,
    ):
        """Add a dense layer to the Gurobi model.

        Parameters
        ----------
        input_vars : mvar_array_like
            Decision variables used as input for the layer
        layer_coefs : np.ndarray
            Weight matrix for the layer
        layer_intercept : np.ndarray
            Bias vector for the layer
        activation : Activation
            Activation function to apply
        activation_vars : mvar, optional
            Output variables for the layer

        Returns
        -------
        DenseLayer
            The created layer object
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
        self,
        input_vars,
        activation,
        activation_vars=None,
        **kwargs,
    ):
        """Add an activation layer to the Gurobi model.

        Parameters
        ----------
        input_vars : mvar_array_like
            Decision variables used as input for the layer
        activation : Activation
            Activation function to apply
        activation_vars : mvar, optional
            Output variables for the layer

        Returns
        -------
        ActivationLayer
            The created layer object
        """
        layer = ActivationLayer(
            self.gp_model,
            activation_vars,
            input_vars,
            activation,
            **kwargs,
        )
        self._layers.append(layer)
        return layer

    def _add_add_layer(
        self,
        input_vars_list,
        output_vars=None,
        **kwargs,
    ):
        """Add an addition layer to the Gurobi model.

        Parameters
        ----------
        input_vars_list : list of mvar
            List of input variable arrays to add together
        output_vars : mvar, optional
            Output variables for the addition

        Returns
        -------
        AddLayer
            The created layer object
        """
        layer = AddLayer(
            self.gp_model,
            output_vars,
            input_vars_list,
            **kwargs,
        )
        self._layers.append(layer)
        return layer

    def get_error(self, eps=None):
        """Compute prediction error.

        This should be implemented by subclasses based on their specific
        predictor type (e.g., ONNX, PyTorch, Keras).
        """
        raise NotImplementedError("Subclasses must implement get_error")
