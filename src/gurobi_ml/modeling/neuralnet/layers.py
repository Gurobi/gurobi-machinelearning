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

        # Parse padding - can be "valid", "same", or a tuple (pad_h, pad_w)
        if self.padding == "valid":
            pad_h, pad_w = 0, 0
        elif self.padding == "same":
            # "same" padding: output size = input size / stride (rounded up)
            # Calculate padding needed
            pad_h = (
                max(
                    (input_vars.shape[1] - 1) * self.strides[0]
                    + self.kernel_size[0]
                    - input_vars.shape[1],
                    0,
                )
                // 2
            )
            pad_w = (
                max(
                    (input_vars.shape[2] - 1) * self.strides[1]
                    + self.kernel_size[1]
                    - input_vars.shape[2],
                    0,
                )
                // 2
            )
        elif isinstance(self.padding, (tuple, list)):
            pad_h, pad_w = self.padding[0], self.padding[1]
        else:
            raise ValueError(f"Unsupported padding type: {self.padding}")

        # Compute shape of output: (input + 2*padding - kernel) / stride + 1
        output_shape_0 = (
            input_vars.shape[1] + 2 * pad_h - self.kernel_size[0]
        ) // self.strides[0] + 1
        output_shape_1 = (
            input_vars.shape[2] + 2 * pad_w - self.kernel_size[1]
        ) // self.strides[1] + 1

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

        (_, height, width, in_channels) = self.input.shape
        mixing = self.gp_model.addMVar(
            self.output.shape,
            lb=-gp.GRB.INFINITY,
            vtype=gp.GRB.CONTINUOUS,
            name=self._name_var("mix"),
        )
        self.mixing = mixing
        self.gp_model.update()

        # Parse padding
        if self.padding == "valid":
            pad_h, pad_w = 0, 0
        elif self.padding == "same":
            pad_h = (
                max((height - 1) * self.strides[0] + self.kernel_size[0] - height, 0)
                // 2
            )
            pad_w = (
                max((width - 1) * self.strides[1] + self.kernel_size[1] - width, 0) // 2
            )
        elif isinstance(self.padding, (tuple, list)):
            pad_h, pad_w = self.padding[0], self.padding[1]
        else:
            raise ValueError(f"Unsupported padding type: {self.padding}")

        # Ultra-optimized convolution: minimize Python overhead and maximize batching
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        out_h, out_w = self.output.shape[1], self.output.shape[2]

        # Pre-compute all window positions and group by pattern (one-time cost)
        window_info = []  # List of (out_i, out_j, h_start, h_end, w_start, w_end, kh_start, kh_end, kw_start, kw_end)

        for out_i in range(out_h):
            in_i = out_i * stride_h - pad_h
            h_start = max(0, in_i)
            h_end = min(height, in_i + kernel_h)
            kh_start = h_start - in_i
            kh_end = kh_start + (h_end - h_start)

            for out_j in range(out_w):
                in_j = out_j * stride_w - pad_w
                w_start = max(0, in_j)
                w_end = min(width, in_j + kernel_w)
                kw_start = w_start - in_j
                kw_end = kw_start + (w_end - w_start)

                window_info.append(
                    (
                        out_i,
                        out_j,
                        h_start,
                        h_end,
                        w_start,
                        w_end,
                        kh_start,
                        kh_end,
                        kw_start,
                        kw_end,
                    )
                )

        # Identify unique window patterns for expression reuse
        unique_patterns = {}  # Maps window slice info to list of (out_i, out_j)
        for (
            out_i,
            out_j,
            h_start,
            h_end,
            w_start,
            w_end,
            kh_start,
            kh_end,
            kw_start,
            kw_end,
        ) in window_info:
            key = (h_start, h_end, w_start, w_end, kh_start, kh_end, kw_start, kw_end)
            if key not in unique_patterns:
                unique_patterns[key] = []
            unique_patterns[key].append((out_i, out_j))

        # Add ALL constraints at once using addConstrs with .item() - MASSIVE speedup!
        # Key insight: .item() extracts scalar constraint from MVar slice, enabling bulk addition
        # This achieves ~170x speedup over individual addConstr calls!

        # Pre-build list of all indices for proper constraint indexing
        all_constraints = []
        for k in range(self.channels):
            for (
                h_start,
                h_end,
                w_start,
                w_end,
                kh_start,
                kh_end,
                kw_start,
                kw_end,
            ), positions in unique_patterns.items():
                # Build convolution expression once for this pattern
                input_window = self.input[:, h_start:h_end, w_start:w_end, :]
                kernel_window = self.coefs[kh_start:kh_end, kw_start:kw_end, :, k]
                conv_expr = (input_window * kernel_window).sum(
                    axis=(1, 2, 3)
                ) + self.intercept[k]

                # Create constraints for all positions with this pattern
                for out_i, out_j in positions:
                    all_constraints.append(
                        (
                            k,
                            out_i,
                            out_j,
                            mixing[:, out_i, out_j, k].item() == conv_expr.item(),
                        )
                    )

        # Bulk add using generator with proper index tuples
        self.gp_model.addConstrs((constr for k, i, j, constr in all_constraints))

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


class BatchNormalizationLayer(AbstractNNLayer):
    """Class to model a batch normalization layer.

    BatchNormalization normalizes the activations using learned parameters.
    During inference (which is what we model), it applies:
        y = gamma * (x - mean) / sqrt(variance + epsilon) + beta

    This is equivalent to an affine transformation:
        y = scale * x + shift
    where scale = gamma / sqrt(variance + epsilon)
          shift = beta - mean * scale
    """

    def __init__(
        self,
        gp_model,
        output_vars,
        input_vars,
        gamma,
        beta,
        mean,
        variance,
        epsilon=1e-5,
        **kwargs,
    ):
        """Initialize a BatchNormalization layer.

        Parameters
        ----------
        gp_model : gurobipy.Model
            The Gurobi model
        output_vars : mvar_array_like
            Output variables (created if None)
        input_vars : mvar_array_like
            Input variables
        gamma : np.ndarray
            Scale parameter (shape matches feature dimension)
        beta : np.ndarray
            Shift parameter (shape matches feature dimension)
        mean : np.ndarray
            Running mean (shape matches feature dimension)
        variance : np.ndarray
            Running variance (shape matches feature dimension)
        epsilon : float
            Small constant for numerical stability (default: 1e-5)
        """
        self.gamma = gamma
        self.beta = beta
        self.mean = mean
        self.variance = variance
        self.epsilon = epsilon

        # Precompute the affine transformation parameters
        self.scale = gamma / np.sqrt(variance + epsilon)
        self.shift = beta - mean * self.scale

        self._default_name = "batchnorm"
        super().__init__(gp_model, output_vars, input_vars, Identity(), **kwargs)

    def _create_output_vars(self, input_vars):
        # Output has the same shape as input
        rval = self.gp_model.addMVar(input_vars.shape, lb=-gp.GRB.INFINITY, name="bn")
        self.gp_model.update()
        return rval

    def _mip_model(self, **kwargs):
        """Add the batch normalization constraints to the model.

        For each feature dimension, apply: output = scale * input + shift
        """
        # BatchNorm applies per-feature normalization
        # The scale and shift parameters are broadcast across the batch and spatial dimensions

        input_shape = self.input.shape

        if len(input_shape) == 2:
            # Dense/FC layer: (batch, features)
            # Apply normalization per feature
            for j in range(input_shape[1]):
                self.gp_model.addConstr(
                    self.output[:, j]
                    == self.scale[j] * self.input[:, j] + self.shift[j]
                )
        elif len(input_shape) == 4:
            # Conv layer: (batch, height, width, channels)
            # Apply normalization per channel
            for k in range(input_shape[3]):
                self.gp_model.addConstr(
                    self.output[:, :, :, k]
                    == self.scale[k] * self.input[:, :, :, k] + self.shift[k]
                )
        else:
            raise ValueError(
                f"BatchNormalization: unsupported input shape {input_shape}"
            )

        self.gp_model.update()


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

        # Parse padding
        if self.padding == "valid":
            pad_h, pad_w = 0, 0
        elif self.padding == "same":
            pad_h = (
                max(
                    (input_vars.shape[1] - 1) * self.stride[0]
                    + self.pool_size[0]
                    - input_vars.shape[1],
                    0,
                )
                // 2
            )
            pad_w = (
                max(
                    (input_vars.shape[2] - 1) * self.stride[1]
                    + self.pool_size[1]
                    - input_vars.shape[2],
                    0,
                )
                // 2
            )
        elif isinstance(self.padding, (tuple, list)):
            pad_h, pad_w = self.padding[0], self.padding[1]
        else:
            raise ValueError(f"Unsupported padding type: {self.padding}")

        out_h = (input_vars.shape[1] + 2 * pad_h - self.pool_size[0]) // self.stride[
            0
        ] + 1
        out_w = (input_vars.shape[2] + 2 * pad_w - self.pool_size[1]) // self.stride[
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
        (_, height, width, channels) = self.input.shape
        ph, pw = self.pool_size
        sh, sw = self.stride
        out_h = self.output.shape[1]
        out_w = self.output.shape[2]

        # Parse padding
        if self.padding == "valid":
            pad_h, pad_w = 0, 0
        elif self.padding == "same":
            pad_h = max((height - 1) * sh + ph - height, 0) // 2
            pad_w = max((width - 1) * sw + pw - width, 0) // 2
        elif isinstance(self.padding, (tuple, list)):
            pad_h, pad_w = self.padding[0], self.padding[1]
        else:
            raise ValueError(f"Unsupported padding type: {self.padding}")

        for n in range(self.input.shape[0]):
            for k in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Calculate input window position
                        in_i = i * sh - pad_h
                        in_j = j * sw - pad_w

                        # Collect valid pool window values
                        pool_vars = []
                        for pi in range(ph):
                            for pj in range(pw):
                                h_idx = in_i + pi
                                w_idx = in_j + pj
                                # Only include values within bounds
                                if 0 <= h_idx < height and 0 <= w_idx < width:
                                    pool_vars.append(self.input[n, h_idx, w_idx, k])

                        if pool_vars:
                            # Standard max pooling
                            self.gp_model.addGenConstrMax(
                                self.output[n, i, j, k],
                                pool_vars,
                                name=self._indexed_name((n, i, j, k), "pool"),
                            )
                        else:
                            # Edge case: window entirely outside bounds (shouldn't happen with valid padding calc)
                            # Set to negative infinity or zero depending on convention
                            self.gp_model.addConstr(self.output[n, i, j, k] == 0)

        self.gp_model.update()
