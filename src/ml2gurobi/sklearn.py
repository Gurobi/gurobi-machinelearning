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

    def __init__(self, model, scaler, input_vars, **kwargs):
        super().__init__(model, input_vars, None, **kwargs)
        self.scaler = scaler

    def transform(self):
        '''Do the transormation on x'''
        _input = self._input

        nfeat = _input.shape[1]
        scale = self.scaler.scale_
        mean = self.scaler.mean_

        begin = self.get_stats_()

        variables = self.model.addMVar(_input.shape, name='__scaledx')
        variables.LB = (_input.LB - mean)/scale
        variables.UB = (_input.UB - mean)/scale
        self.model.addConstrs((_input[:, i] - variables[:, i] * scale[i] == mean[i]
                               for i in range(nfeat)), name='__scaling')
        end = self.get_stats_()
        self.update(begin, end)
        self._output = variables
        return self

    def output(self):
        '''Return the scaled variable features'''
        return self._output


class LinearRegression2Gurobi(BaseNNRegression2Gurobi):
    ''' Predict a Gurobi variable using a Linear Regression that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, model, regressor, input_vars, output_vars, **kwargs):
        super().__init__(model, regressor, input_vars, output_vars, **kwargs)

    def mip_model(self):
        '''Add the prediction constraints to Gurobi'''
        self.addlayer(self._input, self.regressor.coef_.T.reshape(-1, 1),
                      np.array(self.regressor.intercept_).reshape((-1,)),
                      self.actdict['identity'], self._output)


class LogisticRegression2Gurobi(BaseNNRegression2Gurobi):
    ''' Predict a Gurobi variable using a Logistic Regression that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, model, regressor, input_vars, output_vars, **kwargs):
        super().__init__(model, regressor, input_vars, output_vars, **kwargs)

    def mip_model(self):
        '''Add the prediction constraints to Gurobi'''
        self.addlayer(self._input, self.regressor.coef_.T,
                      self.regressor.intercept_, self.actdict['logit'],
                      self._output)


class MLPRegressor2Gurobi(BaseNNRegression2Gurobi):
    ''' Predict a Gurobi matrix variable using a neural network that
        takes another Gurobi matrix variable as input.
        '''

    def __init__(self, model, regressor, input_vars, output_vars, clean_regressor=False, **kwargs):
        super().__init__(model, regressor, input_vars, output_vars,
                         clean_regressor=clean_regressor, **kwargs)
        assert regressor.out_activation_ in ('identity', 'relu', 'softmax')

    def mip_model(self):
        '''Add the prediction constraints to Gurobi'''
        neuralnet = self.regressor
        if neuralnet.activation not in self.actdict:
            print(self.actdict)
            raise BaseException(f'No implementation for activation function {neuralnet.activation}')
        activation = self.actdict[neuralnet.activation]

        input_vars = self._input
        output = None

        for i in range(neuralnet.n_layers_ - 1):
            layer_coefs = neuralnet.coefs_[i]
            layer_intercept = neuralnet.intercepts_[i]

            # For last layer change activation
            if i == neuralnet.n_layers_ -2:
                activation = self.actdict[neuralnet.out_activation_]
                output = self._output

            layer = self.addlayer(input_vars, layer_coefs,
                                  layer_intercept, activation, output,
                                  name=f'{i}')
            input_vars = layer._output  # pylint: disable=W0212
            self.model.update()


class Pipe2Gurobi(MLSubModel):
    '''Use a scikit-learn pipeline to build constraints in Gurobi model.'''
    def __init__(self, model, pipeline, input_vars, output_vars, **kwargs):
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
                self.steps.append(GradientBoostingRegressor2Gurobi(model, obj, None, None,
                                                                   **kwargs))
            else:
                raise BaseException(f"I don't know how to deal with that object: {name}")
        super().__init__(model, input_vars, output_vars, **kwargs)

    def mip_model(self):
        _input = self._input
        for obj in self.steps[:-1]:
            obj._set_input(_input)  # pylint: disable=W0212
            obj.transform()
            _input = obj.output()
        self.steps[-1]._set_input(_input)  # pylint: disable=W0212
        self.steps[-1]._set_output(self._output)  # pylint: disable=W0212
        self.steps[-1]._add() # pylint: disable=W0212

def linearregression2gurobi(model, regressor, X, y, **kwargs):
    ''' Add a linear regression from scikit learn to gurobipy model'''
    return LinearRegression2Gurobi(model, regressor, X, y, **kwargs)


def logisticregression2gurobi(model, regressor, X, y, **kwargs):
    ''' Add a logistic regression from scikit learn to gurobipy model'''
    return LogisticRegression2Gurobi(model, regressor, X, y, **kwargs)


def mlpclassifier2gurobi(model, regressor, X, y, **kwargs):
    ''' Add a neural network classifier from scikit learn to gurobipy model'''
    return MLPRegressor2Gurobi(model, regressor, X, y, **kwargs)


def mlpregressor2gurobi(model, regressor, X, y, **kwargs):
    ''' Add a neural network regression from scikit learn to gurobipy model'''
    return MLPRegressor2Gurobi(model, regressor, X, y, **kwargs)


def decisiontreeregressor2gurobi(model, regressor, X, y, **kwargs):
    ''' Add a decision tree regressor from scikit learn to gurobipy model'''
    return DecisionTree2Gurobi(model, regressor, X, y, **kwargs)


def gradientboostingregressor2gurobi(model, regressor, X, y, **kwargs):
    ''' Add a gradient boosting tree regressor from scikit learn to gurobipy model'''
    return GradientBoostingRegressor2Gurobi(model, regressor, X, y, **kwargs)


def pipe2gurobi(model, pipeline, X, y, **kwargs):
    ''' Add a scikit learn pipeline to gurobipy model'''
    return Pipe2Gurobi(model, pipeline, X, y, **kwargs)
