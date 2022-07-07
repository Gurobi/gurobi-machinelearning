# Copyright © 2022 Gurobi Optimization, LLC

"""Base class for inserting trained predictors into Gurobi
models"""

import gurobipy as gp

# pylint: disable=C0103
import numpy as np

from .activations import Identity, LogitPWL, ReLUGC
from .submodel import SubModel, addtomodel


def validate_gpvars(gpvars):
    """Put variables into appropriate form (matrix of variable)"""
    if isinstance(gpvars, gp.MVar):
        if gpvars.ndim == 1:
            return gp.MVar(gpvars.tolist(), shape=(1, gpvars.shape[0]))
        if gpvars.ndim == 2:
            return gpvars
        raise BaseException("Variables should be an MVar of dimension 1 or 2")
    if isinstance(gpvars, dict):
        gpvars = gpvars.values()
    if isinstance(gpvars, list):
        return gp.MVar(gpvars, shape=(1, len(gpvars)))
    if isinstance(gpvars, gp.Var):
        return gp.MVar(
            [
                gpvars,
            ],
            shape=(1, 1),
        )
    raise BaseException("Could not validate variables")


def transpose(gpvars):
    """Transpose a matrix of variables

    Should I really do this?
    """
    assert isinstance(gpvars, gp.MVar)
    assert gpvars.ndim == 2
    return gp.MVar(gpvars.tolist()[0], (gpvars.shape[1], gpvars.shape[0]))


class AbstractPredictor(SubModel):
    """Class to define a submodel"""

    @addtomodel
    def __init__(self, model, input_vars, *args, output_vars=None, **kwargs):
        assert model == self._model
        self._input = validate_gpvars(input_vars)
        if output_vars is not None:
            self._output = validate_gpvars(output_vars)
        else:
            self._output = None
        self._add(*args, **kwargs)

    def _set_output(self, output_vars):
        self._output = validate_gpvars(output_vars)

    @staticmethod
    def validate(input_vars, output_vars):
        """Validate input and output variables (check shapes, reshape if needed."""
        if input_vars is None:
            raise BaseException("No input variables")
        if output_vars is None:
            raise BaseException("No output variables")

        if (
            output_vars.shape[0] != input_vars.shape[0]
            and output_vars.shape[1] != input_vars.shape[0]
        ):
            raise BaseException(
                "Non-conforming dimension between "
                + "input variable and output variable: "
                + f"{output_vars.shape[0]} != {input_vars.shape[0]}"
            )
        if (
            input_vars.shape[0] != output_vars.shape[0]
            and output_vars.shape[1] == input_vars.shape[0]
        ):
            output_vars = transpose(output_vars)

        return (input_vars, output_vars)

    def _add(self, *args, **kwargs):
        """Predict output from input using regression/classifier"""
        if self._output is None:
            self._create_output_vars(self._input, *args, **kwargs)
        self._input, self._output = self.validate(self._input, self._output)
        self.mip_model(*args, **kwargs)
        return self

    def _create_output_vars(self, input_vars):
        """Impemented in child classes. Does nothing in the abstract class."""
        assert input_vars is not None
        assert self._model is not None
        assert False

    def mip_model(self, activation=None):
        """Impemented in child classes. Does nothing in the abstract class."""
        assert self._model is not None
        assert activation is None or activation is not None
        assert False


class NNLayer(AbstractPredictor):
    """Class to build one layer of a neural network"""

    def __init__(
        self,
        model,
        output_vars,
        input_vars,
        layer_coefs,
        layer_intercept,
        activation_function,
        *args,
        **kwargs,
    ):
        self.coefs = layer_coefs
        self.intercept = layer_intercept
        self.activation = activation_function
        self.wmin = None
        self.wmax = None
        self.zvar = None
        super().__init__(model, input_vars, output_vars, *args, **kwargs)

    def _wminmax(self):
        """Compute min/max for w variable"""
        if (self._input.UB >= gp.GRB.INFINITY).any():
            return (
                -gp.GRB.INFINITY * np.ones(self._output.shape),
                gp.GRB.INFINITY * np.ones(self._output.shape),
            )
        if (self._input.LB <= -gp.GRB.INFINITY).any():
            return (
                -gp.GRB.INFINITY * np.ones(self._output.shape),
                gp.GRB.INFINITY * np.ones(self._output.shape),
            )
        wpos = np.maximum(self.coefs, 0.0)
        wneg = np.minimum(self.coefs, 0.0)
        wmin = self._input.LB @ wpos + self._input.UB @ wneg + self.intercept
        wmax = self._input.UB @ wpos + self._input.LB @ wneg + self.intercept
        wmax = np.maximum(wmin, wmax)

        return (wmin, wmax)

    def _create_output_vars(self, input_vars):
        rval = self._model.addMVar(
            (input_vars.shape[0], self.coefs.shape[1]), lb=-gp.GRB.INFINITY, name="act"
        )
        self._model.update()
        return rval

    def mip_model(self, activation=None):
        """Add the layer to model"""
        model = self._model
        model.update()
        if activation is None:
            activation = self.activation

        # Compute bounds on weighted sums by propagation
        wmin, wmax = self._wminmax()

        # Take best bound from what we have stored and what we propagated
        if wmax is not None and self.wmax is not None:
            wmax = np.minimum(wmax, self.wmax)
        if wmin is not None and self.wmin is not None:
            wmin = np.maximum(wmin, self.wmin)
        self.wmin = wmin
        self.wmax = wmax

        # Do the mip model for the activation in the layer
        activation.mip_model(self)
        self._model.update()

    def reset_bounds(self):
        """Reset bounds on layer"""
        activation_function = self.activation
        self._model.update()
        activation_function.reset_bounds(self)
        self._model.update()

    def _update(self, model, before):
        """Update added modeling objects compared to status before."""
        model.update()
        # range of variables
        if model.numVars > before.numVars:
            self._firstVar = model.getVars()[before.numVars]
            self._lastVar = model.getVars()[model.numVars - 1]
        # range of constraints
        if model.numConstrs > before.numConstrs:
            self._firstConstr = model.getConstrs()[before.numConstrs]
            self._lastConstr = model.getConstrs()[model.numConstrs - 1]
        # range of Q constraints
        if model.numQConstrs > before.numQConstrs:
            self._QConstrs = model.getQConstrs()[before.numQConstrs : model.numQConstrs]
        # range of GenConstrs
        if model.numGenConstrs > before.numGenConstrs:
            self._GenConstrs = model.getGenConstrs()[
                before.numGenConstrs : model.numGenConstrs
            ]
        # range of SOS
        if model.numSOS > before.numSOS:
            self._SOSs = model.getSOSs()[before.numSOS : model.numSOS]

    def redolayer(self, activation=None):
        """Rebuild the layer (possibly using a different model for activation)"""
        self._model.remove(self.getConstrs())
        self._model.remove(self.getQConstrs())
        self._model.remove(self.getGenConstrs())
        self._model.update()
        before = SubModel._modelstats(self._model)
        self._add(activation)
        self._update(self._model, before)


class BaseNNPredictor(AbstractPredictor):
    """Base class for inserting a regressor based on neural-network/tensor into Gurobi"""

    def __init__(
        self, model, regressor, input_vars, output_vars, clean_regressor=False, **kwargs
    ):
        self.regressor = regressor
        self.clean = clean_regressor
        self.actdict = {"relu": ReLUGC(), "identity": Identity(), "logit": LogitPWL()}
        self._layers = []
        super().__init__(model, input_vars, output_vars, **kwargs)

    def __iter__(self):
        return self._layers.__iter__()

    def addlayer(
        self,
        input_vars,
        layer_coefs,
        layer_intercept,
        activation,
        activation_vars=None,
        name=None,
    ):
        """Add a layer to model"""
        if self.clean:
            mask = np.abs(layer_coefs) < 1e-8
            layer_coefs[mask] = 0.0
        layer = NNLayer(
            self._model,
            activation_vars,
            input_vars,
            layer_coefs,
            layer_intercept,
            activation,
            name,
        )
        self._layers.append(layer)
        return layer

    def rebuild_formulation(self, activation=None):
        """Rebuild the MIP formulation for regression model"""
        for layer in self:
            if not isinstance(layer.activation, Identity):
                layer.redolayer(activation)
        self._model.update()

    def reset_bounds(self):
        """Reset bounds of variables in mip model"""
        for layer in self:
            layer.reset_bounds()
        self._model.update()
