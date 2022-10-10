# Copyright © 2022 Gurobi Optimization, LLC

""" To transform a sequential neural network of PyTorch
in a Gurobi model """


from tensorflow import keras

from ..modeling.neuralnet import BaseNNConstr


class KerasNetworkConstr(BaseNNConstr):
    def __init__(self, grbmodel, predictor, input_vars, output_vars, clean_regressor=False):
        assert predictor.built
        for step in predictor.layers:
            if isinstance(step, keras.layers.Dense):
                config = step.get_config()
                if config["activation"] not in ("relu", "linear"):
                    raise Exception("Unsupported network structure")
            elif isinstance(step, keras.layers.ReLU):
                pass
            elif isinstance(step, keras.layers.InputLayer):
                pass
            else:
                raise Exception("Unsupported network structure")

        super().__init__(grbmodel, predictor, input_vars, output_vars, clean_regressor=clean_regressor)

    def mip_model(self):
        network = self.predictor
        _input = self._input
        output = None
        numlayers = len(network.layers)

        for i, step in enumerate(network.layers):
            if i == numlayers - 1:
                output = self._output
            if isinstance(step, keras.layers.InputLayer):
                pass
            elif isinstance(step, keras.layers.ReLU):
                layer = self.add_activation_layer(
                    _input,
                    self.actdict["relu"],
                    output,
                    name=f"{i}",
                )
                _input = layer.output
            else:
                config = step.get_config()
                activation = config["activation"]
                if activation == "linear":
                    activation = "identity"
                weights, bias = step.get_weights()
                layer = self.add_dense_layer(
                    _input,
                    weights,
                    bias,
                    self.actdict[activation],
                    output,
                    name=f"{i}",
                )
                _input = layer.output

    def get_error(self):
        if self.has_solution():
            return self.predictor.predict(self.input.X) - self.output.X
        raise BaseException("No solution available")


def add_keras_constr(grbmodel, keras_model, input_vars, output_vars=None, **kwargs):
    return KerasNetworkConstr(grbmodel, keras_model, input_vars, output_vars, **kwargs)
