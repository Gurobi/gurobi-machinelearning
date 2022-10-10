# Copyright © 2022 Gurobi Optimization, LLC
"""Bases classes for modeling neural network layers"""

import numpy as np

from ..modeling.basepredictor import AbstractPredictorConstr, _default_name
from .activations import Identity, LogisticGC, ReLUGC
from .layers import ActivationLayer, DenseLayer


class BaseNNConstr(AbstractPredictorConstr):
    """Base class for inserting a regressor based on neural-network/tensor into Gurobi"""

    def __init__(self, grbmodel, predictor, input_vars, output_vars, clean_regressor=False, **kwargs):
        self.predictor = predictor
        self.clean = clean_regressor
        self.actdict = {"relu": ReLUGC(), "identity": Identity(), "logistic": LogisticGC()}
        try:
            for activation, activation_model in kwargs["activation_models"].items():
                self.actdict[activation] = activation_model
        except KeyError:
            pass
        self._layers = []

        default_name = kwargs.pop("default_name", _default_name(predictor))
        super().__init__(
            grbmodel=grbmodel, input_vars=input_vars, output_vars=output_vars, default_name=default_name, **kwargs
        )

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
