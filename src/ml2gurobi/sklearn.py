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
from .utils import MLSubModel


class StandardScaler2Gurobi(MLSubModel):
    ''' Class to use a StandardScale to create scaled version of
        some Gurobi variables. '''

    def __init__(self, model, scaler, input, **kwargs):
        super().__init__(model, input, None, **kwargs)
        self.scaler = scaler

    def transform(self):
        '''Do the transormation on x'''
        X = self._input

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
        self.output_ = variables
        return self

    def output(self):
        '''Return the scaled variable features'''
        return self.output_


class LinearRegression2Gurobi(BaseNNRegression2Gurobi):
    ''' Predict a Gurobi variable using a Linear Regression that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, model, regressor, input, output, **kwargs):
        super().__init__(model, regressor, input, output, **kwargs)

    def mip_model(self):
        '''Add the prediction constraints to Gurobi'''
        self.addlayer(self._input, self.regressor.coef_.T.reshape(-1, 1),
                      np.array(self.regressor.intercept_).reshape((-1,)),
                      self.actdict['identity'], self._output)


class LogisticRegression2Gurobi(BaseNNRegression2Gurobi):
    ''' Predict a Gurobi variable using a Logistic Regression that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, model, regressor, input, output, **kwargs):
        super().__init__(model, regressor, input, output, **kwargs)

    def mip_model(self):
        '''Add the prediction constraints to Gurobi'''
        self.addlayer(self._input, self.regressor.coef_.T,
                      self.regressor.intercept_, self.actdict['logit'],
                      self._output)


class MLPRegressor2Gurobi(BaseNNRegression2Gurobi):
    ''' Predict a Gurobi matrix variable using a neural network that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, model, regressor, input, output, clean_regressor=False, **kwargs):
        super().__init__(model, regressor, input, output,
                         clean_regressor=clean_regressor, **kwargs)
        assert regressor.out_activation_ in ('identity', 'relu', 'softmax')

    def mip_model(self):
        '''Add the prediction constraints to Gurobi'''
        neuralnet = self.regressor
        if neuralnet.activation not in self.actdict:
            print(self.actdict)
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
            activations = layer._output
            self.model.update()


class Pipe2Gurobi(MLSubModel):
    '''Use a scikit-learn pipeline to build constraints in Gurobi model.'''
    def __init__(self, model, pipeline, input, output, **kwargs):
        super().__init__(model, input, output)
        self.steps = []
        for name, obj in pipeline.steps:
            if name == 'standardscaler':
                self.steps.append(StandardScaler2Gurobi(model, obj, None, **kwargs))
            elif name == 'linearregression':
                self.steps.append(LinearRegression2Gurobi(model, obj, None, None, **kwargs))
            elif name == 'logisticregression':
                self.steps.append(LogisticRegression2Gurobi(model, obj, None, None, **kwargs))
            elif name == 'mlpregressor':
                self.steps.append(MLPRegressor2Gurobi(model, obj, None, None, **kwargs))
            elif name == 'mlpclassifier':
                self.steps.append(MLPRegressor2Gurobi(model, obj, None, None, **kwargs))
            elif name == 'decisiontreeregressor':
                self.steps.append(DecisionTree2Gurobi(model, obj, None, None, **kwargs))
            elif name == 'gradientboostingregressor':
                self.steps.append(GradientBoostingRegressor2Gurobi(model, obj, None, None, **kwargs))
            else:
                raise BaseException(f"I don't know how to deal with that object: {name}")

    def mip_model(self):
        X = self._input
        for obj in self.steps[:-1]:
            obj._set_input(X)
            obj.transform()
            X = obj.output()
        self.steps[-1]._set_input(X)
        self.steps[-1]._set_output(self._output)
        self.steps[-1].predict()

def add_linearregression(model, regressor, X, y, **kwargs):
    return LinearRegression2Gurobi(model, regressor, X, y, **kwargs).predict()


def add_logisticregression(model, regressor, X, y, **kwargs):
    return LogisticRegression2Gurobi(model, regressor, X, y, **kwargs).predict()


def add_mlpclassifier(model, regressor, X, y, **kwargs):
    return MLPRegressor2Gurobi(model, regressor, X, y, **kwargs).predict()


def add_mlpregressor(model, regressor, X, y, **kwargs):
    return MLPRegressor2Gurobi(model, regressor, X, y, **kwargs).predict()


def add_decisiontreeregressor(model, regressor, X, y, **kwargs):
    return DecisionTree2Gurobi(model, regressor, X, y, **kwargs).predict()


def add_gradientboostingregressor(model, regressor, X, y, **kwargs):
    return GradientBoostingRegressor2Gurobi(model, regressor, X, y, **kwargs).predict()


def add_pipe(model, pipe, X, y, **kwargs):
    return Pipe2Gurobi(model, pipe, X, y, **kwargs).predict()
