# Copyright © 2022 Gurobi Optimization, LLC
"""Bases classes for modeling neural network layers"""

import gurobipy as gp

from ..base.basepredictor import AbstractPredictorConstr, _default_name
from ..base.submodel import SubModel


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
        AbstractPredictorConstr.__init__(self, grbmodel, input_vars, output_vars, **kwargs)

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
        self.zvar = None
        super().__init__(grbmodel, output_vars, input_vars, activation_function, **kwargs)

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

        # Do the mip model for the activation in the layer
        activation.mip_model(self)
        self._model.update()
