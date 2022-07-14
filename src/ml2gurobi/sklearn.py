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
            for i, power in enumerate(powers):
                qexpr = gp.QuadExpr()
                qexpr += 1.0
                for j, feat in enumerate(_input[k, :]):
                    if power[j] == 2:
                        qexpr *= feat
                        qexpr *= feat
                    elif power[j] == 1:
                        qexpr *= feat
                self.model.addConstr(output[k, i] == qexpr, name=f"polyfeat[{k},{i}]")


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
            self._output = self._layers[-1].output


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
            self._output = self._layers[-1].output


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

            layer = self.addlayer(input_vars, layer_coefs, layer_intercept, activation, output, name=f"layer{i}")
            input_vars = layer._output  # pylint: disable=W0212
            self._model.update()
        if self._output is None:
            self._output = layer.output


class PipelineConstr(AbstractPredictorConstr):
    """Use a scikit-learn pipeline to build constraints in Gurobi model."""

    def __init__(self, grbmodel, pipeline, input_vars, output_vars=None, **kwargs):
        self._steps = []
        self.pipeline = pipeline
        self._kwargs = kwargs
        super().__init__(grbmodel, input_vars, output_vars, **kwargs)

    def mip_model(self):
        pipeline = self.pipeline
        model = self._model
        input_vars = self._input
        output_vars = self._output
        steps = self._steps
        for name, obj in pipeline.steps[:-1]:
            if name == "standardscaler":
                steps.append(StandardScalerConstr(model, obj, input_vars, **self._kwargs))
            elif name == "polynomialfeatures":
                steps.append(PolynomialFeaturesConstr(model, obj, input_vars, **self._kwargs))
            else:
                raise BaseException(f"I don't know how to deal with that object: {name}")
            input_vars = steps[-1].output
        name, obj = pipeline.steps[-1]
        if name == "linearregression":
            steps.append(LinearRegressionConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "logisticregression":
            steps.append(LogisticRegressionConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "mlpregressor":
            steps.append(MLPRegressorConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "mlpclassifier":
            steps.append(MLPRegressorConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "decisiontreeregressor":
            steps.append(DecisionTreeRegressorConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "gradientboostingregressor":
            steps.append(GradientBoostingRegressorConstr(model, obj, input_vars, output_vars, **self._kwargs))
        elif name == "randomforestregressor":
            steps.append(RandomForestRegressorConstr(model, obj, input_vars, output_vars, **self._kwargs))
        else:
            raise BaseException(f"I don't know how to deal with that object: {name}")
        if self._output is None:
            self._output = steps[-1].output

    def __getitem__(self, key):
        return self._steps[key]

    def __iter__(self):
        return self._steps.__iter__()

    def __len__(self):
        return self._steps.__len__()
