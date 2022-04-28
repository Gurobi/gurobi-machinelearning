# Copyright © 2022 Gurobi Optimization, LLC
''' A set of classes for transforming Scikit-Learn objects
to constraints in Gurobi

This tries to be very simple. Each object is constructed by passing
in a scikit-learn object we want to transform and the model we want
to transform it into.

What we have so far:
  - StandardScaler2Grb: would create scaled version of a set of variables
    in a Gurobi model: xscale = scaled(x)
  - LinearRegression2Grb: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - LogisticRegression2Grb: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - MLSRegression2Grb: a neural network.
  - Pipe2Gurobi: convert a scikit-learn pipeline.

'''

# pylint: disable=C0103

import numpy as np

from .ml2grb import BaseNNRegression2Grb
from .utils import validate_gpvars


class StandardScaler2Grb:
    ''' Class to use a StandardScale to create scaled version of
        some Gurobi variables. '''

    def __init__(self, scaler, model, **kwargs):
        self.scaler_ = scaler
        self.model_ = model
        self.constrs_ = None
        self.vars_ = None

    def transform(self, X):
        '''Do the transormation on x'''
        self.model_.update()

        X = validate_gpvars(X)

        nfeat = X.shape[1]
        scale = self.scaler_.scale_
        mean = self.scaler_.mean_

        self.vars_ = self.model_.addMVar(X.shape, name='__scaledx')
        self.vars_.LB = (X.LB - mean)/scale
        self.vars_.UB = (X.UB - mean)/scale
        self.constrs_ = [self.model_.addConstr(X[:, i] - self.vars_[:, i] * scale[i] == mean[i],
                                               name=f'__scaling[{i}]')
                         for i in range(nfeat)]
        return self

    def X(self):
        '''Return the scaled variable features'''
        return self.vars_


class LinearRegression2Grb(BaseNNRegression2Grb):
    ''' Predict a Gurobi variable using a Linear Regression that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, regressor, model, **kwargs):
        BaseNNRegression2Grb.__init__(self, regressor, model, **kwargs)

    def predict(self, X, y):
        '''Add the prediction constraints to Gurobi'''
        X, y = self.validate(X, y)
        self.addlayer(X, self.regressor.coef_.T.reshape(-1, 1),
                      np.array(self.regressor.intercept_).reshape((-1,)), self.actdict['identity'], y)
        return self


class LogisticRegression2Grb(BaseNNRegression2Grb):
    ''' Predict a Gurobi variable using a Logistic Regression that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, regressor, model, **kwargs):
        BaseNNRegression2Grb.__init__(self, regressor, model, **kwargs)

    def predict(self, X, y):
        '''Add the prediction constraints to Gurobi'''
        X, y = self.validate(X, y)
        self.addlayer(X, self.regressor.coef_.T,
                      self.regressor.intercept_, self.actdict['logit'], y)
        return self


class MLPRegressor2Grb(BaseNNRegression2Grb):
    ''' Predict a Gurobi matrix variable using a neural network that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, regressor, model, clean_regressor=False, **kwargs):
        BaseNNRegression2Grb.__init__(self, regressor, model, clean_regressor, **kwargs)
        assert regressor.out_activation_ in ('identity', 'relu', 'softmax')

    def predict(self, X, y):
        '''Add the prediction constraints to Gurobi'''
        neuralnet = self.regressor
        if neuralnet.activation not in self.actdict:
            raise BaseException(f'No implementation for activation function {neuralnet.activation}')
        activation = self.actdict[neuralnet.activation]

        X, y = self.validate(X, y)
        self._input = X
        self._output = y

        activations = X
        # Iterate over the layers
        for i in range(neuralnet.n_layers_ - 1):
            layer_coefs = neuralnet.coefs_[i]
            layer_intercept = neuralnet.intercepts_[i]

            input_vars = activations

            # For last layer change activation
            if i == neuralnet.n_layers_ - 2:
                activations = y
                if neuralnet.out_activation_ != neuralnet.activation:
                    activation = self.actdict[neuralnet.out_activation_]
            else:
                activations = None

            layer = self.addlayer(input_vars, layer_coefs,
                                  layer_intercept, activation, activations,
                                  name=f'{i}')
            activations = layer.actvar
            self.model.update()


class Pipe2Gurobi:
    '''Use a scikit-learn pipeline to build constraints in Gurobi model.'''
    def __init__(self, pipeline, model, **kwargs):
        self.model = model
        self.steps = []
        for name, obj in pipeline.steps:
            if name == 'standardscaler':
                self.steps.append(StandardScaler2Grb(obj, model, **kwargs))
            elif name == 'linearregression':
                self.steps.append(LinearRegression2Grb(obj, model, **kwargs))
            elif name == 'logisticregression':
                self.steps.append(LogisticRegression2Grb(obj, model, **kwargs))
            elif name == 'mlpregressor':
                self.steps.append(MLPRegressor2Grb(obj, model, **kwargs))
            elif name == 'mlpclassifier':
                self.steps.append(MLPRegressor2Grb(obj, model, **kwargs))
            else:
                raise BaseException(f"I don't know how to deal with that object: {obj.__name__}")

    def predict(self, X, y):
        for obj in self.steps[:-1]:
            obj.transform(X)
            X = obj.X()
        self.steps[-1].predict(X, y)
        return self


def pipe_predict(X, y, pipe, model, **kwargs):
    return Pipe2Gurobi(pipe, model, **kwargs).predict(X, y)
