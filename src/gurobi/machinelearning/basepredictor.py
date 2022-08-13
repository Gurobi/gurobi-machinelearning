# Copyright © 2022 Gurobi Optimization, LLC

"""Base class for inserting trained predictors into Gurobi
models"""

import gurobipy as gp
import numpy as np

from .activations import Identity, LogitPWL, ReLUGC
from .submodel import SubModel


def _default_name(predictor):
    """Make a default name for predictor constraint."""
    return type(predictor).__name__.lower()


def validate_gpvars(gpvars, isinput):
    """Put variables into appropriate form (matrix of variable)"""
    if isinstance(gpvars, gp.MVar):
        if gpvars.ndim == 1 and isinput:
            return gp.MVar(gpvars.tolist(), shape=(1, gpvars.shape[0]))
        if gpvars.ndim in (1, 2):
            return gpvars
        raise BaseException("Variables should be an MVar of dimension 1 or 2")
    if isinstance(gpvars, dict):
        gpvars = gpvars.values()
    if isinstance(gpvars, list):
        if isinput:
            return gp.MVar(gpvars, shape=(1, len(gpvars)))
        return gp.MVar(gpvars, shape=(len(gpvars)))
    if isinstance(gpvars, gp.Var):
        return gp.MVar(
            [
                gpvars,
            ],
            shape=(1, 1),
        )
    raise BaseException("Could not validate variables")


class AbstractPredictorConstr(SubModel):
    """Class to define a submodel"""

    def __init__(self, grbmodel, input_vars, output_vars=None, **kwargs):
        self._input = validate_gpvars(input_vars, True)
        if output_vars is not None:
            self._output = validate_gpvars(output_vars, False)
        else:
            self._output = None
        super().__init__(grbmodel, **kwargs)

    def _set_output(self, output_vars):
        self._output = validate_gpvars(output_vars, False)

    @staticmethod
    def validate(input_vars, output_vars):
        """Validate input and output variables (check shapes, reshape if needed."""
        if input_vars is None:
            raise BaseException("No input variables")
        if output_vars is None:
            raise BaseException("No output variables")
        if output_vars.ndim == 1:
            if input_vars.shape[0] == 1:
                output_vars = gp.MVar(output_vars.tolist(), shape=(1, output_vars.shape[0]))
            else:
                output_vars = gp.MVar(output_vars.tolist(), shape=(output_vars.shape[0], 1))

        if output_vars.shape[0] != input_vars.shape[0]:
            raise BaseException(
                "Non-conforming dimension between "
                + "input variable and output variable: "
                + f"{output_vars.shape[0]} != {input_vars.shape[0]}"
            )

        return (input_vars, output_vars)

    def _build_submodel(self, model, *args, **kwargs):
        """Predict output from input using regression/classifier"""
        if self._output is None:
            self._create_output_vars(self._input)
        if self._output is not None:
            self._input, self._output = self.validate(self._input, self._output)
        else:
            self._input = validate_gpvars(self._input, True)
        self.mip_model()
        assert self._output is not None
        return self

    def print_stats(self, file=None):
        super().print_stats(file)
        print(f"Input has shape {self.input.shape}", file=file)
        print(f"Output has shape {self.output.shape}", file=file)

    def mip_model(self):
        """Defined in derived class the mip_model for the predictor"""

    def _create_output_vars(self, input_vars):
        """May be defined in derived class to create the output variables of predictor"""
        try:
            n_outputs = self.n_outputs_
        except AttributeError:
            return
        rval = self._model.addMVar((input_vars.shape[0], n_outputs), lb=-gp.GRB.INFINITY, name="output")
        self._model.update()
        self._output = rval

    @property
    def output(self):
        return self._output

    @property
    def input(self):
        return self._input


class AbstractNNLayer(AbstractPredictorConstr):
    """Abstract class for NN layers"""

    def __init__(
        self,
        grbmodel,
        output_vars,
        input_vars,
        activation_function,
        **kwargs,
    ):
        self.activation = activation_function
        super().__init__(grbmodel, input_vars, output_vars, **kwargs)

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
        if model.numvars > before.numvars:
            self._firstvar = model.getVars()[before.numVars]
            self._lastvar = model.getVars()[model.numVars - 1]
        # range of constraints
        if model.numconstrs > before.numconstrs:
            self._firstconstr = model.getConstrs()[before.numconstrs]
            self._lastconstr = model.getConstrs()[model.numconstrs - 1]
        # range of Q constraints
        if model.numqconstrs > before.numqconstrs:
            self._qconstrs = model.getQConstrs()[before.numQConstrs : model.numQConstrs]
        # range of GenConstrs
        if model.numgenconstrs > before.numgenconstrs:
            self._genconstrs = model.getGenConstrs()[before.numgenconstrs : model.numgenconstrs]
        # range of SOS
        if model.numsos > before.numsos:
            self._sos = model.getSOSs()[before.numsos : model.numsos]

    def redolayer(self, activation=None):
        """Rebuild the layer (possibly using a different model for activation)"""
        self.model.remove(self.constrs)
        self.model.remove(self.qconstrs)
        self.model.remove(self.genconstrs)
        self.model.remove(self.sos)
        self.model.update()
        before = SubModel._modelstats(self.model)
        self.mip_model(activation)
        self._update(self.model, before)

    def print_stats(self, file=None):
        """Print statistics about submodel created"""
        super().print_stats(file)
        print(f"Activation is {_default_name(self.activation)}")


class ActivationLayer(AbstractNNLayer):
    """Class to build one layer of a neural network"""

    def __init__(
        self,
        grbmodel,
        output_vars,
        input_vars,
        activation_function,
        **kwargs,
    ):
        self.zvar = None
        super().__init__(grbmodel, output_vars, input_vars, activation_function, **kwargs)

    def _create_output_vars(self, input_vars):
        rval = self._model.addMVar(input_vars.shape, lb=-gp.GRB.INFINITY, name="act")
        self._model.update()
        self._output = rval

    def mip_model(self, activation=None):
        """Add the layer to model"""
        model = self._model
        model.update()
        if activation is None:
            activation = self.activation

        # Do the mip model for the activation in the layer
        activation.mip_model(self)
        self._model.update()


class DenseLayer(AbstractNNLayer):
    """Class to build one layer of a neural network"""

    def __init__(
        self,
        grbmodel,
        output_vars,
        input_vars,
        layer_coefs,
        layer_intercept,
        activation_function,
        **kwargs,
    ):
        self.coefs = layer_coefs
        self.intercept = layer_intercept
        self.wmin = None
        self.wmax = None
        self.zvar = None
        super().__init__(grbmodel, output_vars, input_vars, activation_function, **kwargs)

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
        rval = self._model.addMVar((input_vars.shape[0], self.coefs.shape[1]), lb=-gp.GRB.INFINITY, name="act")
        self._model.update()
        self._output = rval

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


class BaseNNConstr(AbstractPredictorConstr):
    """Base class for inserting a regressor based on neural-network/tensor into Gurobi"""

    def __init__(self, grbmodel, regressor, input_vars, output_vars, clean_regressor=False, **kwargs):
        self.regressor = regressor
        self.clean = clean_regressor
        self.actdict = {"relu": ReLUGC(), "identity": Identity(), "logit": LogitPWL()}
        try:
            for activation, activation_model in kwargs["activation_models"].items():
                self.actdict[activation] = activation_model
        except KeyError:
            pass
        self._layers = []
        super().__init__(grbmodel, input_vars, output_vars, default_name=_default_name(regressor), **kwargs)

    def __iter__(self):
        return self._layers.__iter__()

    def add_dense_layer(self, input_vars, layer_coefs, layer_intercept, activation, activation_vars=None, **kwargs):
        """Add a layer to model"""
        if self.clean:
            mask = np.abs(layer_coefs) < 1e-8
            layer_coefs[mask] = 0.0
        layer = DenseLayer(self._model, activation_vars, input_vars, layer_coefs, layer_intercept, activation, **kwargs)
        self._layers.append(layer)
        return layer

    def add_activation_layer(self, input_vars, activation, activation_vars=None, **kwargs):
        """Add a layer to model"""
        layer = ActivationLayer(self._model, activation_vars, input_vars, activation, **kwargs)
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

    def print_stats(self, file=None):
        """Print statistics about submodel created"""
        name = self._name
        if name == "":
            name = self.default_name
        super().print_stats(file)
        print(file=file)
        print(f"{name} has {len(self._layers)} layers:")
        for layer in self:
            layer.print_stats(file)
            print(file=file)
