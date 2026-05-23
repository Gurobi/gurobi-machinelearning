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

import io

import gurobipy as gp
import numpy as np

from gurobi_ml.modeling.neuralnet.activations import Identity

from ...exceptions import ModelConfigurationError
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
        self.layer_decomposition = kwargs.get("layer_decomposition", True)
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def _build_submodel(self, gp_model, *args, **kwargs):
        self.layer_decomposition = kwargs.get("layer_decomposition", True)
        if not self.layer_decomposition:
            self._mip_model(**kwargs)
            return self
        return super()._build_submodel(gp_model, *args, **kwargs)


    def get_error(self, eps=None):
        # We can't compute externally the error of a layer
        raise NotImplementedError("get_error is not supported for individual NN layers")

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
        if not self.layer_decomposition:
            if not hasattr(self.activation, "_nl_expr"):
                raise ModelConfigurationError(
                    self.activation,
                    f"layer_decomposition=False is not supported with activation '{type(self.activation).__name__}'.",
                )
            if self._output is not None:
                shape = self._output.shape
                for index in np.ndindex(shape):
                    self.gp_model.addConstr(
                        self._output[index] == self.activation._nl_expr(self._input[index]),
                        name=self._indexed_name(index, "full_net"),
                    )
            else:
                shape = self._input.shape
                next_expr = np.empty(shape, dtype=object)
                for index in np.ndindex(shape):
                    next_expr[index] = self.activation._nl_expr(self._input[index])
                self._output = next_expr
            return

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
        if not self.layer_decomposition:
            if not hasattr(self.activation, "_nl_expr"):
                raise ModelConfigurationError(
                    self.activation,
                    f"layer_decomposition=False is not supported with activation '{type(self.activation).__name__}'.",
                )
            linear_expr = self._input @ self.coefs + self.intercept
            if self._output is not None:
                shape = self._output.shape
                for index in np.ndindex(shape):
                    self.gp_model.addConstr(
                        self._output[index] == self.activation._nl_expr(linear_expr[index]),
                        name=self._indexed_name(index, "full_net"),
                    )
            else:
                shape = linear_expr.shape
                next_expr = np.empty(shape, dtype=object)
                for index in np.ndindex(shape):
                    next_expr[index] = self.activation._nl_expr(linear_expr[index])
                self._output = next_expr
            return

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
