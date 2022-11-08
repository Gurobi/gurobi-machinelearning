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

import gurobipy as gp

from ..basepredictor import AbstractPredictorConstr, _default_name


class AbstractNNLayer(AbstractPredictorConstr):
    """Abstract class for NN layers"""

    def __init__(
        self,
        gp_model,
        output_vars,
        input_vars,
        activation_function,
        **kwargs,
    ):
        self.activation = activation_function
        AbstractPredictorConstr.__init__(self, gp_model, input_vars, output_vars, **kwargs)

    def print_stats(self, file=None):
        """Print statistics about submodel created

        Parameters
        ---------

        file: None, optional
          Text stream to which output should be redirected. By default sys.stdout.
        """
        print(
            f"{self._name:12} {_default_name(self.activation):12} {self.output.shape.__str__():>12} {len(self.vars):>10} {len(self.constrs):>10} {len(self.qconstrs):>10} {len(self.genconstrs):>10}",
            file=file,
        )


class ActivationLayer(AbstractNNLayer):
    """Class to build one activation layer of a neural network"""

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
        rval = self._gp_model.addMVar(input_vars.shape, lb=-gp.GRB.INFINITY, name="act")
        self._gp_model.update()
        self._output = rval

    def _mip_model(self, **kwargs):
        """Add the layer to model"""
        model = self.gp_model
        model.update()
        if "activation" in kwargs:
            activation = kwargs["activation"]
        else:
            activation = self.activation

        # Do the mip model for the activation in the layer
        activation.mip_model(self)
        self._gp_model.update()


class DenseLayer(AbstractNNLayer):
    """Class to build one layer of a neural network"""

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
        rval = self._gp_model.addMVar(
            (input_vars.shape[0], self.coefs.shape[1]), lb=-gp.GRB.INFINITY, name="act"
        )
        self._gp_model.update()
        self._output = rval

    def _mip_model(self, **kwargs):
        """Add the layer to model"""
        model = self.gp_model
        model.update()
        if "activation" in kwargs:
            activation = kwargs["activation"]
        else:
            activation = self.activation

        # Do the mip model for the activation in the layer
        activation.mip_model(self)
        self._gp_model.update()
