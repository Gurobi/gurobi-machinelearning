# Copyright © 2022 Gurobi Optimization, LLC
''' A set of classes for transforming Scikit-Learn objects
to constraints in Gurobi

This tries to be very simple. Each object is constructed by passing
in a scikit-learn object we want to transform and the model we want
to transform it into.

What we have so far:
  - StandardScaler2Gurobi: would create scaled version of a set of variables
    in a Gurobi model: xscale = scaled(x)
  - LinearRegression2Gurobi: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - LogisticRegression2Gurobi: insert a constraint of the form y = g(x, psi)
    where g is the regressor prediticted by a logitstic regression.
  - MLSRegression2Gurobi: a neural network.
  - Pipe2Gurobi: convert a scikit-learn pipeline.

'''

# pylint: disable=C0103

import numpy as np

from .decisiontrees import DecisionTree2Gurobi, GradientBoostingRegressor2Gurobi
from .nnbase import BaseNNRegression2Gurobi
from .utils import Submodel, validate_gpvars


class StandardScaler2Gurobi(Submodel):
    ''' Class to use a StandardScale to create scaled version of
        some Gurobi variables. '''

    def __init__(self, scaler, model, **kwargs):
        super().__init__(model, **kwargs)
        self.scaler = scaler
        self.vars_ = None

    def transform(self, X):
        '''Do the transormation on x'''
        X = validate_gpvars(X)

        nfeat = X.shape[1]
        scale = self.scaler.scale_
        mean = self.scaler.mean_

        begin = self.get_stats_()

        variables = self.model.addMVar(X.shape, name='__scaledx')
        variables.LB = (X.LB - mean)/scale
        variables.UB = (X.UB - mean)/scale
        self.model.addConstrs((X[:, i] - variables[:, i] * scale[i] == mean[i]
                               for i in range(nfeat)), name='__scaling')
        end = self.get_stats_()
        self.update(begin, end)
        self.vars_ = variables
        return self

    def X(self):
        '''Return the scaled variable features'''
        return self.vars_


class LinearRegression2Gurobi(BaseNNRegression2Gurobi):
    ''' Predict a Gurobi variable using a Linear Regression that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, regressor, model, **kwargs):
        super().__init__(regressor, model, **kwargs)

    def mip_model(self):
        '''Add the prediction constraints to Gurobi'''
        self.addlayer(self._input, self.regressor.coef_.T.reshape(-1, 1),
                      np.array(self.regressor.intercept_).reshape((-1,)),
                      self.actdict['identity'], self._output)


class LogisticRegression2Gurobi(BaseNNRegression2Gurobi):
    ''' Predict a Gurobi variable using a Logistic Regression that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, regressor, model, **kwargs):
        super().__init__(regressor, model, **kwargs)

    def mip_model(self):
        '''Add the prediction constraints to Gurobi'''
        self.addlayer(self._input, self.regressor.coef_.T,
                      self.regressor.intercept_, self.actdict['logit'],
                      self._output)


class MLPRegressor2Gurobi(BaseNNRegression2Gurobi):
    ''' Predict a Gurobi matrix variable using a neural network that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, regressor, model, clean_regressor=False, **kwargs):
        super().__init__(regressor, model, clean_regressor=clean_regressor, **kwargs)
        assert regressor.out_activation_ in ('identity', 'relu', 'softmax')

    def mip_model(self):
        '''Add the prediction constraints to Gurobi'''
        neuralnet = self.regressor
        if neuralnet.activation not in self.actdict:
            raise BaseException(f'No implementation for activation function {neuralnet.activation}')
        activation = self.actdict[neuralnet.activation]

        activations = self._input
        for i in range(neuralnet.n_layers_ - 1):
            layer_coefs = neuralnet.coefs_[i]
            layer_intercept = neuralnet.intercepts_[i]

            input_vars = activations

            # For last layer change activation
            if i == neuralnet.n_layers_ - 2:
                activations = self._output
                if neuralnet.out_activation_ != neuralnet.activation:
                    activation = self.actdict[neuralnet.out_activation_]
            else:
                activations = None

            layer = self.addlayer(input_vars, layer_coefs,
                                  layer_intercept, activation, activations,
                                  name=f'{i}')
            activations = layer.actvar
            self.model.update()


class Pipe2Gurobi(Submodel):
    '''Use a scikit-learn pipeline to build constraints in Gurobi model.'''
    def __init__(self, pipeline, model, **kwargs):
        super().__init__(model)
        self.steps = []
        for name, obj in pipeline.steps:
            if name == 'standardscaler':
                self.steps.append(StandardScaler2Gurobi(obj, model, **kwargs))
            elif name == 'linearregression':
                self.steps.append(LinearRegression2Gurobi(obj, model, **kwargs))
            elif name == 'logisticregression':
                self.steps.append(LogisticRegression2Gurobi(obj, model, **kwargs))
            elif name == 'mlpregressor':
                self.steps.append(MLPRegressor2Gurobi(obj, model, **kwargs))
            elif name == 'mlpclassifier':
                self.steps.append(MLPRegressor2Gurobi(obj, model, **kwargs))
            elif name == 'decisiontreeregressor':
                self.steps.append(DecisionTree2Gurobi(obj, model, **kwargs))
            elif name == 'gradientboostingregressor':
                self.steps.append(GradientBoostingRegressor2Gurobi(obj, model, **kwargs))
            else:
                raise BaseException(f"I don't know how to deal with that object: {name}")

    def mip_model(self):
        X = self._input
        for obj in self.steps[:-1]:
            obj.transform(X)
            X = obj.X()
        self.steps[-1].predict(X, self._output)

def add_linearregression(regressor, model, X, y, **kwargs):
    return LinearRegression2Gurobi(regressor, model, **kwargs).predict(X, y)

def add_logisticregression(regressor, model, X, y, **kwargs):
    return LogisticRegression2Gurobi(regressor, model, **kwargs).predict(X, y)

def add_mlpclassifier(regressor, model, X, y, **kwargs):
    return MLPRegressor2Gurobi(regressor, model, **kwargs).predict(X, y)

def add_mlpregressor(regressor, model, X, y, **kwargs):
    return MLPRegressor2Gurobi(regressor, model, **kwargs).predict(X, y)

def add_decisiontreeregressor(regressor, model, X, y, **kwargs):
    return DecisionTree2Gurobi(regressor, model, **kwargs).predict(X, y)

def add_gradientboostingregressor(regressor, model, X, y, **kwargs):
    return GradientBoostingRegressor2Gurobi(regressor, model, **kwargs).predict(X, y)

def add_pipe(pipe, model, X, y, **kwargs):
    return Pipe2Gurobi(pipe, model, **kwargs).predict(X, y)
