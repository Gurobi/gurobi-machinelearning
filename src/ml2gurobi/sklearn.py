# Copyright © 2022 Gurobi Optimization, LLC
""" A set of classes for transforming Scikit-Learn objects
to constraints in Gurobi

This tries to be very simple. Each object is constructed by passing
in a scikit-learn object we want to transform and the model we want
to transform it into.

What we have so far:
  - StandardScalerConstr: would create scaled version of a set of variables
    in a Gurobi model: xscale = scaled(x)
  - PolynomialFeaturesTransform: creates monomials from input variables.
  - LinearRegressionConstr: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - LogisticRegressionConstr: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - MLSRegressionConstr: a neural network.
  - PipeConstr: convert a scikit-learn pipeline.

"""

# pylint: disable=C0103

import gurobipy as gp
import numpy as np

from .basepredictor import AbstractPredictorConstr, BaseNNConstr
from .decisiontrees import (
    DecisionTreeRegressorConstr,
    GradientBoostingRegressorConstr,
    RandomForestRegressorConstr,
)


class StandardScalerConstr(AbstractPredictorConstr):
    """Class to use a StandardScale to create scaled version of
    some Gurobi variables."""

    def __init__(self, grbmodel, scaler, input_vars, **kwargs):
        self.scaler = scaler
        super().__init__(grbmodel, input_vars, **kwargs)

    def _create_output_vars(self, input_vars, **kwargs):
        rval = self._model.addMVar(input_vars.shape, name="scaledx")
        self._model.update()
        self._output = rval

    def mip_model(self):
        """Do the transormation on x"""
        _input = self._input
        output = self._output

        nfeat = _input.shape[1]
        scale = self.scaler.scale_
        mean = self.scaler.mean_

        output.LB = (_input.LB - mean) / scale
        output.UB = (_input.UB - mean) / scale
        self._model.addConstrs(
            (_input[:, i] - output[:, i] * scale[i] == mean[i] for i in range(nfeat)),
            name="s",
        )
        return self

    def output(self):
        """Return the scaled variable features"""
        return self._output


class PolynomialFeaturesConstr(AbstractPredictorConstr):
    """Class to use a PolynomialFeatures to create transforms of
    some Gurobi variables."""

    def __init__(self, grbmodel, polytrans, input_vars, **kwargs):
        if polytrans.degree > 2:
            raise BaseException("Can only handle polynomials of degree < 2")
        self.polytrans = polytrans
        super().__init__(grbmodel, input_vars, **kwargs)

    def _create_output_vars(self, input_vars, **kwargs):
        out_shape = (input_vars.shape[0], self.polytrans.n_output_features_)
        rval = self._model.addMVar(out_shape, name="polyx", lb=-gp.GRB.INFINITY)
        self._model.update()
        self._output = rval

    def mip_model(self):
        """Do the transormation on x"""
        _input = self._input
        output = self._output

        nexamples, nfeat = _input.shape
        powers = self.polytrans.powers_
        assert powers.shape[0] == self.polytrans.n_output_features_
        assert powers.shape[1] == nfeat

        for k in range(nexamples):
            for i, p in enumerate(powers):
                q = gp.QuadExpr()
                q += 1.0
                for j, feat in enumerate(_input[k, :]):
                    if p[j] == 2:
                        q *= feat
                        q *= feat
                    elif p[j] == 1:
                        q *= feat
                self.model.addConstr(output[k, i] == q, name=f"polyfeat[{k},{i}]")

    def output(self):
        """Return the scaled variable features"""
        return self._output


class LinearRegressionConstr(BaseNNConstr):
    """Predict a Gurobi variable using a Linear Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, regressor, input_vars, output_vars=None, **kwargs):
        super().__init__(grbmodel, regressor, input_vars, output_vars, **kwargs)

    def mip_model(self):
        """Add the prediction constraints to Gurobi"""
        self.addlayer(
            self._input,
            self.regressor.coef_.T.reshape(-1, 1),
            np.array(self.regressor.intercept_).reshape((-1,)),
            self.actdict["identity"],
            self._output,
        )
        if self._output is None:
            self._output = self._layers[-1]._output


class LogisticRegressionConstr(BaseNNConstr):
    """Predict a Gurobi variable using a Logistic Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, regressor, input_vars, output_vars=None, **kwargs):
        super().__init__(grbmodel, regressor, input_vars, output_vars, **kwargs)

    def mip_model(self):
        """Add the prediction constraints to Gurobi"""
        self.addlayer(
            self._input,
            self.regressor.coef_.T,
            self.regressor.intercept_,
            self.actdict["logit"],
            self._output,
        )
        if self._output is None:
            self._output = self._layers[-1]._output


class MLPRegressorConstr(BaseNNConstr):
    """Predict a Gurobi matrix variable using a neural network that
    takes another Gurobi matrix variable as input.
    """

    def __init__(self, grbmodel, regressor, input_vars, output_vars=None, clean_regressor=False, **kwargs):
        super().__init__(
            grbmodel,
            regressor,
            input_vars,
            output_vars,
            clean_regressor=clean_regressor,
            **kwargs,
        )
        assert regressor.out_activation_ in ("identity", "relu", "softmax")

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

            layer = self.addlayer(input_vars, layer_coefs, layer_intercept, activation, output, name=f"layer{i}")
            input_vars = layer._output  # pylint: disable=W0212
            self._model.update()
        if self._output is None:
            self._output = layer._output


class PipelineConstr(AbstractPredictorConstr):
    """Use a scikit-learn pipeline to build constraints in Gurobi model."""

    def __init__(self, grbmodel, pipeline, input_vars, output_vars=None, **kwargs):
        self.steps = []
        self.pipeline = pipeline
        self._kwargs = kwargs
        super().__init__(grbmodel, input_vars, output_vars, **kwargs)

    def mip_model(self):
        pipeline = self.pipeline
        model = self._model
        input_vars = self._input
        output_vars = self._output
        for name, obj in pipeline.steps[:-1]:
            if name == "standardscaler":
                self.steps.append(StandardScalerConstr(model, obj, input_vars, **self._kwargs))
            elif name == "polynomialfeatures":
                self.steps.append(PolynomialFeaturesConstr(model, obj, input_vars, **self._kwargs))
            else:
                raise BaseException(f"I don't know how to deal with that object: {name}")
            input_vars = self.steps[-1].output()
        name, obj = pipeline.steps[-1]
        if name == "linearregression":
            self.steps.append(LinearRegressionConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "logisticregression":
            self.steps.append(LogisticRegressionConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "mlpregressor":
            self.steps.append(MLPRegressorConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "mlpclassifier":
            self.steps.append(MLPRegressorConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "decisiontreeregressor":
            self.steps.append(DecisionTreeRegressorConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "gradientboostingregressor":
            self.steps.append(GradientBoostingRegressorConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "randomforestregressor":
            self.steps.append(RandomForestRegressorConstr(model, obj, input_vars, output_vars, **self._kwargs))
        else:
            raise BaseException(f"I don't know how to deal with that object: {name}")
        if self._output is None:
            self._output = self.steps[-1]._output
