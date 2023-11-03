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

"""Bases classes for modeling neural network layers."""

import io

import gurobipy as gp

from gurobi_ml.modeling.neuralnet.activations import Identity

from .._var_utils import _default_name
from ..base_predictor_constr import AbstractPredictorConstr


class AbstractNNLayer(AbstractPredictorConstr):
    """Abstract class for NN layers."""

    def __init__(
        self,
        gp_model,
        output_vars,
        input_vars,
        activation_function,
        **kwargs,
    ):
        self.activation = activation_function
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def get_error(self, eps=None):
        # We can't compute externally the error of a layer
        assert False

    def print_stats(self, abbrev=False, file=None):
        """Print statistics about submodel created.

        Parameters
        ----------

        file : None, optional
          Text stream to which output should be redirected. By default sys.stdout.
        """
        return AbstractPredictorConstr.print_stats(self, True, file)


class ActivationLayer(AbstractNNLayer):
    """Class to build one activation layer of a neural network."""

    def __init__(
        self,
        gp_model,
        output_vars,
        input_vars,
        activation_function,
        **kwargs,
    ):
        self.zvar = None
        self._default_name = "activation"
        super().__init__(
            gp_model,
            output_vars,
            input_vars,
            activation_function,
            **kwargs,
        )

    def _create_output_vars(self, input_vars):
        rval = self.gp_model.addMVar(input_vars.shape, lb=-gp.GRB.INFINITY, name="act")
        self.gp_model.update()
        return rval

    def _mip_model(self, **kwargs):
        """Add the layer to model."""
        model = self.gp_model
        model.update()
        if "activation" in kwargs:
            activation = kwargs["activation"]
        else:
            activation = self.activation

        # Do the mip model for the activation in the layer
        activation.mip_model(self)
        self.gp_model.update()


class DenseLayer(AbstractNNLayer):
    """Class to build one layer of a neural network."""

    def __init__(
        self,
        gp_model,
        output_vars,
        input_vars,
        layer_coefs,
        layer_intercept,
        activation_function,
        **kwargs,
    ):
        self.coefs = layer_coefs
        self.intercept = layer_intercept
        self.zvar = None
        self._default_name = "dense"
        super().__init__(
            gp_model,
            output_vars,
            input_vars,
            activation_function,
            **kwargs,
        )

    def _create_output_vars(self, input_vars):
        rval = self.gp_model.addMVar(
            (input_vars.shape[0], self.coefs.shape[1]), lb=-gp.GRB.INFINITY, name="act"
        )
        self.gp_model.update()
        return rval

    def _mip_model(self, **kwargs):
        """Add the layer to model."""
        model = self.gp_model
        model.update()
        if "activation" in kwargs:
            activation = kwargs["activation"]
        else:
            activation = self.activation

        # Do the mip model for the activation in the layer
        activation.mip_model(self)
        self.gp_model.update()

    def print_stats(self, abbrev=False, file=None):
        """Print statistics about submodel created.

        Parameters
        ----------

        file : None, optional
          Text stream to which output should be redirected. By default sys.stdout.
        """
        if not isinstance(self.activation, Identity):
            output = io.StringIO()
            AbstractPredictorConstr.print_stats(self, abbrev=True, file=output)
            activation_name = f"({_default_name(self.activation)})"

            out_string = output.getvalue()
            print(f"{out_string[:-1]} {activation_name}", file=file)
            return
        AbstractPredictorConstr.print_stats(self, abbrev=True, file=file)
