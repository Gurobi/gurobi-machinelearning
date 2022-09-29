# Copyright © 2022 Gurobi Optimization, LLC
""" A set of classes for transforming Scikit-Learn regression objects
to constraints in Gurobi

What we have so far:
  - LinearRegressionConstr: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - LogisticRegressionConstr: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - MLSRegressionConstr: a neural network.
"""


import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..basepredictor import BaseNNConstr


class LinearRegressionConstr(BaseNNConstr):
    """Predict a Gurobi variable using a Linear Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, regressor, input_vars, output_vars=None, **kwargs):
        check_is_fitted(regressor)
        super().__init__(grbmodel, regressor, input_vars, output_vars, **kwargs)

    def mip_model(self):
        """Add the prediction constraints to Gurobi"""
        self.add_dense_layer(
            self._input,
            self.regressor.coef_.T.reshape(-1, 1),
            np.array(self.regressor.intercept_).reshape((-1,)),
            self.actdict["identity"],
            self._output,
        )
        if self._output is None:
            self._output = self._layers[-1].output


class LogisticRegressionConstr(BaseNNConstr):
    """Predict a Gurobi variable using a Logistic Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, regressor, input_vars, output_vars=None, **kwargs):
        check_is_fitted(regressor)
        super().__init__(grbmodel, regressor, input_vars, output_vars, **kwargs)

    def mip_model(self):
        """Add the prediction constraints to Gurobi"""
        self.add_dense_layer(
            self._input,
            self.regressor.coef_.T,
            self.regressor.intercept_,
            self.actdict["logistic"],
            self._output,
        )
        if self._output is None:
            self._output = self._layers[-1].output


class MLPRegressorConstr(BaseNNConstr):
    """Predict a Gurobi matrix variable using a neural network that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, regressor, input_vars, output_vars=None, clean_regressor=False, **kwargs):
        check_is_fitted(regressor)
        super().__init__(
            grbmodel,
            regressor,
            input_vars,
            output_vars,
            clean_regressor=clean_regressor,
            **kwargs,
        )
        assert regressor.out_activation_ in ("identity", "relu")

    def mip_model(self):
        """Add the prediction constraints to Gurobi"""
        neuralnet = self.regressor
        if neuralnet.activation not in self.actdict:
            print(self.actdict)
            raise BaseException(f"No implementation for activation function {neuralnet.activation}")
        activation = self.actdict[neuralnet.activation]

        input_vars = self._input
        output = None

        for i in range(neuralnet.n_layers_ - 1):
            layer_coefs = neuralnet.coefs_[i]
            layer_intercept = neuralnet.intercepts_[i]

            # For last layer change activation
            if i == neuralnet.n_layers_ - 2:
                activation = self.actdict[neuralnet.out_activation_]
                output = self._output

            layer = self.add_dense_layer(input_vars, layer_coefs, layer_intercept, activation, output, name=f"layer{i}")
            input_vars = layer._output  # pylint: disable=W0212
            self._model.update()
        if self._output is None:
            self._output = layer.output
