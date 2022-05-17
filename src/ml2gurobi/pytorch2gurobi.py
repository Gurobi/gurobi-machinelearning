# Copyright © 2022 Gurobi Optimization, LLC
# pylint: disable=C0103

from torch import nn

from .ml2gurobi import BaseNNRegression2Gurobi


class Sequential2Gurobi(BaseNNRegression2Gurobi):
    '''Transform a pytorch Sequential Neural Network to Gurboi constraint with
       input and output as matrices of variables.'''
    def __init__(self, regressor, model, clean_regressor=False):
        BaseNNRegression2Gurobi.__init__(self, regressor, model, clean_regressor)

        linear = None
        for step in regressor:
            if isinstance(step, nn.ReLU):
                assert linear is not None
                linear = None
            elif isinstance(step, nn.Linear):
                assert linear is None
                linear = step
            else:
                print(step)
                raise BaseException("Unsupported network structure")

    def predict(self, X, y):
        network = self.regressor
        X, y = self.validate(X, y)
        self._input = X
        self._output = y
        input_vars = X
        linear = None
        for i, step in enumerate(network):
            if isinstance(step, nn.ReLU):
                for name, param in linear.named_parameters():
                    if name == 'weight':
                        layer_weight = param.detach().numpy().T
                    elif name == 'bias':
                        layer_bias = param.detach().numpy()
                layer = self.addlayer(input_vars, layer_weight,
                                      layer_bias, self.actdict['relu'], None, name=f'{i}')
                linear = None
                input_vars = layer.actvar
            elif isinstance(step, nn.Linear):
                assert linear is None
                linear = step
            else:
                raise BaseException("Unsupported network structure")
        if linear is not None:
            for name, param in linear.named_parameters():
                if name == 'weight':
                    layer_weight = param.detach().numpy().T
                elif name == 'bias':
                    layer_bias = param.detach().numpy()
            self.addlayer(input_vars, layer_weight, layer_bias, self.actdict['identity'], y)
