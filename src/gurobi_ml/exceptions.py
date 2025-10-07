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

"""Exceptions for gurobi_ml.

This module defines the exception hierarchy used throughout gurobi-ml.
All exceptions inherit from appropriate Python standard exceptions to provide
clear semantic meaning and consistent behavior.
"""


class GurobiMLError(Exception):
    """Base class for all gurobi-ml exceptions."""


class PredictorNotSupportedError(NotImplementedError, GurobiMLError):
    """Predictor type is not supported by gurobi-ml.

    Raised when attempting to use a predictor type that hasn't been registered
    or implemented in gurobi-ml.

    Parameters
    ----------
    predictor : str or object
        The predictor type that is not supported
    """

    def __init__(self, predictor):
        predictor_name = (
            predictor if isinstance(predictor, str) else type(predictor).__name__
        )
        super().__init__(
            f"Predictor type '{predictor_name}' is not supported by gurobi-ml. "
            f"Check the documentation for supported predictor types or consider "
            f"registering a custom converter."
        )


class ModelConfigurationError(ValueError, GurobiMLError):
    """Predictor configuration cannot be converted to optimization model.

    Raised when a predictor's structure, parameters, or configuration
    prevents it from being formulated as a mathematical optimization model.

    Parameters
    ----------
    predictor : str or object
        The predictor that has the configuration issue
    reason : str
        Description of why the configuration is not supported
    """

    def __init__(self, predictor, reason):
        predictor_name = (
            predictor if isinstance(predictor, str) else type(predictor).__name__
        )
        super().__init__(
            f"Cannot create optimization model for {predictor_name}: {reason}"
        )


class NoSolutionError(RuntimeError, GurobiMLError):
    """Gurobi model has no solution available.

    Raised when attempting to retrieve solution values from a Gurobi model
    that doesn't have a solution (not optimized, infeasible, unbounded, etc.).

    Parameters
    ----------
    message : str, optional
        Custom error message. If not provided, uses a default message.
    """

    def __init__(self, message=None):
        if message is None:
            message = "No solution available from Gurobi model"
        super().__init__(message)


# Legacy aliases - deprecated, will be removed in future versions
# These are provided temporarily to help with migration
class NotRegistered(PredictorNotSupportedError):
    """Deprecated: Use PredictorNotSupportedError instead."""

    def __init__(self, predictor):
        import warnings

        warnings.warn(
            "NotRegistered is deprecated and will be removed. Use PredictorNotSupportedError instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(predictor)


class NoModel(ModelConfigurationError):
    """Deprecated: Use ModelConfigurationError instead."""

    def __init__(self, predictor, reason):
        import warnings

        warnings.warn(
            "NoModel is deprecated and will be removed. Use ModelConfigurationError instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(predictor, reason)


class NoSolution(NoSolutionError):
    """Deprecated: Use NoSolutionError instead."""

    def __init__(self):
        import warnings

        warnings.warn(
            "NoSolution is deprecated and will be removed. Use NoSolutionError instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__("No solution available")


class ParameterError(ValueError, GurobiMLError):
    """Deprecated: Use standard ValueError instead."""

    def __init__(self, message):
        import warnings

        warnings.warn(
            "ParameterError is deprecated and will be removed. Use ValueError instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(message)
