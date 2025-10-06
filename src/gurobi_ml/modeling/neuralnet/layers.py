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
import numpy as np

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

        mixing = self.gp_model.addMVar(
            self.output.shape,
            lb=-gp.GRB.INFINITY,
            vtype=gp.GRB.CONTINUOUS,
            name=self._name_var("mix"),
        )
        self.mixing = mixing
        self.gp_model.update()

        self.gp_model.addConstr(self.mixing == self.input @ self.coefs + self.intercept)
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


class Conv2DLayer(AbstractNNLayer):
    """Class to build one convolution 2D layer of a neural network."""

    def __init__(
        self,
        gp_model,
        output_vars,
        input_vars,
        layer_coefs,
        layer_intercept,
        channels,
        kernel_size,
        stride,
        padding,
        activation_function,
        **kwargs,
    ):
        self.coefs = layer_coefs
        self.intercept = layer_intercept
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = stride
        self.padding = padding
        self.zvar = None
        self._default_name = "conv2d"
        super().__init__(
            gp_model,
            output_vars,
            input_vars,
            activation_function,
            **kwargs,
        )

    def _create_output_vars(self, input_vars):
        assert len(input_vars.shape) == 4

        # compute shape of output
        # should be (input + padding)/stride
        padding = 0
        output_shape = input_vars.shape[1] + 2 * padding - self.kernel_size[0]
        output_shape /= self.strides[0]
        output_shape += 1
        output_shape_0 = output_shape
        output_shape = input_vars.shape[2] + 2 * padding - self.kernel_size[1]
        output_shape /= self.strides[1]
        output_shape += 1
        output_shape_1 = output_shape
        output_shape = (
            input_vars.shape[0],
            int(output_shape_0),
            int(output_shape_1),
            self.channels,
        )
        print(
            f"Conv2D layer with input shape {input_vars.shape} gives output shape {output_shape}"
        )
        print(
            f"  kernel size {self.kernel_size}, stride {self.strides}, padding {self.padding}"
        )
        rval = self.gp_model.addMVar(output_shape, lb=-gp.GRB.INFINITY, name="act")
        self.gp_model.update()
        return rval

    def _mip_model(self, **kwargs):
        """Add the layer to model."""
        model = self.gp_model
        model.update()

        (_, height, width, _) = self.input.shape
        mixing = self.gp_model.addMVar(
            self.output.shape,
            lb=-gp.GRB.INFINITY,
            vtype=gp.GRB.CONTINUOUS,
            name=self._name_var("mix"),
        )
        self.mixing = mixing
        self.gp_model.update()

        assert self.padding == "valid"

        # Here comes the complicated loop...
        # I am sure there is a better way but this is a pedestrian version
        kernel_w, kernel_h = self.kernel_size
        stride_h, stride_w = self.strides
        for k in range(self.channels):
            for out_i, i in enumerate(range(0, height - kernel_h + 1, stride_h)):
                if i + kernel_h > height:
                    continue
                for out_j, j in enumerate(range(0, width - kernel_w + 1, stride_w)):
                    if j + kernel_w > width:
                        continue
                    self.gp_model.addConstr(
                        mixing[:, out_i, out_j, k]
                        == (
                            self.input[:, i : i + kernel_h, j : j + kernel_w, :]
                            * self.coefs[:, :, :, k]
                        ).sum()
                        + self.intercept[k]
                    )

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


class FlattenLayer(AbstractNNLayer):
    """Class to flatten the output of a convolutional neural network."""

    def __init__(self, gp_model, output_vars, input_vars, **kwargs):
        self._default_name = "flatten"
        super().__init__(gp_model, output_vars, input_vars, Identity(), **kwargs)

    def _create_output_vars(self, input_vars):
        assert len(input_vars.shape) >= 2
        output_shape = (input_vars.shape[0], int(np.prod(input_vars.shape[1:])))
        print(f"Flattening {input_vars.shape} into {output_shape}")
        rval = self.gp_model.addMVar(output_shape, lb=-gp.GRB.INFINITY, name="act")
        self.gp_model.update()
        return rval

    def _mip_model(self, **kwargs):
        # Mark kwargs as used to avoid unused argument warning
        _ = kwargs
        # Flatten the input explicitly to match the output shape using a pedestrian approach
        input_shape = self.input.shape
        batch_size = input_shape[0]
        for n in range(batch_size):
            flat_idx = 0
            for idx in np.ndindex(input_shape[1:]):
                self.gp_model.addConstr(
                    self.output[n, flat_idx] == self.input[(n,) + idx]
                )
                flat_idx += 1


class MaxPooling2DLayer(AbstractNNLayer):
    """Class to model a max pooling 2D layer."""

    def __init__(
        self,
        gp_model,
        output_vars,
        input_vars,
        pool_size,
        stride,
        padding,
        **kwargs,
    ):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self._default_name = "maxpool2d"
        super().__init__(gp_model, output_vars, input_vars, Identity(), **kwargs)

    def _create_output_vars(self, input_vars):
        assert len(input_vars.shape) == 4
        pad = 0
        out_h = (input_vars.shape[1] + 2 * pad - self.pool_size[0]) // self.stride[
            0
        ] + 1
        out_w = (input_vars.shape[2] + 2 * pad - self.pool_size[1]) // self.stride[
            1
        ] + 1
        output_shape = (
            input_vars.shape[0],
            int(out_h),
            int(out_w),
            input_vars.shape[3],
        )
        rval = self.gp_model.addMVar(output_shape, lb=-gp.GRB.INFINITY, name="act")
        self.gp_model.update()
        print(
            f"MaxPool2D layer with input shape {input_vars.shape} gives output shape {output_shape}"
        )
        return rval

    def _mip_model(self, **kwargs):
        assert self.padding == "valid"
        (_, height, width, channels) = self.input.shape
        ph, pw = self.pool_size
        sh, sw = self.stride
        out_h = self.output.shape[1]
        out_w = self.output.shape[2]
        for n in range(self.input.shape[0]):
            for k in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        ii = i * sh
                        jj = j * sw
                        pool_vars = self.input[
                            n, ii : ii + ph, jj : jj + pw, k
                        ].reshape(-1)
                        self.gp_model.addGenConstrMax(
                            self.output[n, i, j, k],
                            pool_vars.tolist(),
                            name=self._indexed_name((n, i, j, k), "pool"),
                        )
        self.gp_model.update()
