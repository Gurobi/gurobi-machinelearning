# Copyright © 2022 Gurobi Optimization, LLC

""" Module for inserting a Keras model into a gurobipy model
"""


from tensorflow import keras

from ..exceptions import NoModel, NoSolution
from ..modeling.neuralnet import BaseNNConstr


class KerasNetworkConstr(BaseNNConstr):
    def __init__(self, grbmodel, predictor, input_vars, output_vars, **kwargs):
        assert predictor.built
        for step in predictor.layers:
            if isinstance(step, keras.layers.Dense):
                config = step.get_config()
                if config["activation"] not in ("relu", "linear"):
                    raise NoModel(predictor, "Unsupported network structure")
            elif isinstance(step, keras.layers.ReLU):
                pass
            elif isinstance(step, keras.layers.InputLayer):
                if step.negative_slope != 0.0:
                    raise NoModel(predictor, "Only handle ReLU layers with negative slope 0.0")
                if step.threshold != 0.0:
                    raise NoModel(predictor, "Only handle ReLU layers with threshold of 0.0")
                if step.max_value is not None and step.max_value < float("inf"):
                    raise NoModel(predictor, "Only handle ReLU layers without maxvalue")
            else:
                raise NoModel(predictor, "Unsupported network structure")

        super().__init__(grbmodel, predictor, input_vars, output_vars, **kwargs)

    def _mip_model(self):
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
        """Returns error in Gurobi's solution with respect to prediction from input

        Returns
        -------
        float
            Assuming that we have a solution for the input and output variables
            `x, y`. Returns the difference between `predict(x)` and
            `y`, where predict is the corresponding function for the Scikit-Learn
            object we are modeling.

        Raises
        ------
        NoSolution
            If the Gurobi model has no solution (either was not optimized or is infeasible).
        """
        if self._has_solution():
            return self.predictor.predict(self.input.X) - self.output.X
        raise NoSolution()


def add_keras_constr(grbmodel, keras_model, input_vars, output_vars=None, **kwargs):
    """Use `keras_model` to predict the value of `output_vars` using `input_vars` in `grbmodel`

    Parameters
    ----------
    grbmodel: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
        The gurobipy model where the predictor should be inserted.
    keras_model: `keras.Model <https://keras.io/api/models/model/>`
        The keras model to insert as predictor.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    KerasNetworkConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Raises
    ------
    NoModel
        If the translation for some of the Keras model structure
        (layer or activation) is not implemented.

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars
    """
    return KerasNetworkConstr(grbmodel, keras_model, input_vars, output_vars, **kwargs)
