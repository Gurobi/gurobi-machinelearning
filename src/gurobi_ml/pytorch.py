# Copyright © 2022 Gurobi Optimization, LLC
# pylint: disable=C0103

""" To transform a sequential neural network of PyTorch
in a Gurobi model """

from torch import nn

from .basepredictor import BaseNNConstr


class Sequential(BaseNNConstr):
    """Transform a pytorch Sequential Neural Network to Gurboi constraint with
    input and output as matrices of variables."""

    def __init__(self, grbmodel, regressor, input_vars, output_vars, clean_regressor=False):
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
        BaseNNConstr.__init__(
            self, grbmodel, regressor, input_vars, output_vars, clean_regressor=clean_regressor, default_name="torchsequential"
        )

    def mip_model(self):
        network = self.regressor
        _input = self._input
        output = self._output

        linear = None
        for i, step in enumerate(network):
            if isinstance(step, nn.ReLU):
                for name, param in linear.named_parameters():
                    if name == "weight":
                        layer_weight = param.detach().numpy().T
                    elif name == "bias":
                        layer_bias = param.detach().numpy()
                layer = self.add_dense_layer(
                    _input,
                    layer_weight,
                    layer_bias,
                    self.actdict["relu"],
                    None,
                    name=f"{i}",
                )
                linear = None
                _input = layer.output
            elif isinstance(step, nn.Linear):
                assert linear is None
                linear = step
            else:
                raise BaseException("Unsupported network structure")
        if linear is not None:
            for name, param in linear.named_parameters():
                if name == "weight":
                    layer_weight = param.detach().numpy().T
                elif name == "bias":
                    layer_bias = param.detach().numpy()
            self.add_dense_layer(_input, layer_weight, layer_bias, self.actdict["identity"], output)
