# Copyright © 2022 Gurobi Optimization, LLC
# pylint: disable=C0103

""" Module for inserting a :external+torch:py:class:`torch.nn.Sequential` model into a gurobipy model
"""

import torch
from torch import nn

from ..exceptions import NoModel, NoSolution
from ..modeling.neuralnet import BaseNNConstr


class SequentialConstr(BaseNNConstr):
    """Transform a pytorch Sequential Neural Network to Gurobi constraint with
    input and output as matrices of variables."""

    def __init__(self, grbmodel, predictor, input_vars, output_vars, **kwargs):
        linear = None
        for step in predictor:
            if isinstance(step, nn.ReLU):
                assert linear is not None
                linear = None
            elif isinstance(step, nn.Linear):
                assert linear is None
                linear = step
            else:
                print(step)
                raise NoModel(predictor, "Unsupported network structure")
        super().__init__(grbmodel, predictor, input_vars, output_vars, default_name="torchsequential")

    def _mip_model(self):
        network = self.predictor
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
                raise NoModel(network, "Unsupported network structure")
        if linear is not None:
            for name, param in linear.named_parameters():
                if name == "weight":
                    layer_weight = param.detach().numpy().T
                elif name == "bias":
                    layer_bias = param.detach().numpy()
            self.add_dense_layer(_input, layer_weight, layer_bias, self.actdict["identity"], output)

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
            t_in = torch.from_numpy(self.input.X).float()
            t_out = self.predictor.forward(t_in)
            return t_out.detach().numpy() - self.output.X
        raise NoSolution()


def add_sequential_constr(grbmodel, sequential_model, input_vars, output_vars=None, **kwargs):
    """Use a `sequential_model` to predict the value of `output_vars` using `input_vars` in `grbmodel`

    Parameters
    ----------
    grbmodel: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
        The gurobipy model where the predictor should be inserted.
    sequential_model: :external+torch:py:class:`torch.nn.Sequential`
        The sequential model to insert as predictor.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    SequentialConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Raises
    ------
    NoModel
        If the translation for some of the Pytorch model structure
        (layer or activation) is not implemented.

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars
    """
    return SequentialConstr(grbmodel, sequential_model, input_vars, output_vars, **kwargs)
