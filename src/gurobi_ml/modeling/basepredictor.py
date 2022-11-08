# Copyright © 2022 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import gurobipy as gp

from ..exceptions import ParameterError
from .submodel import SubModel


def _default_name(predictor):
    """Make a default name for predictor constraint.

    Parameters
    ----------
    predictor:
        Class of the predictor
    """
    return type(predictor).__name__.lower()


def validate_gp_vars(gp_vars, is_input):
    """Put variables into appropriate form (matrix of variable).

    Parameters
    ----------
    gpvars:
        Decision variables used.
    isinput:
        True if variables are used as input. False if variables are used
        as output.

    Returns
    -------
    mvar_array_like
        Decision variables with correctly adjusted shape.
    """
    if isinstance(gp_vars, gp.MVar):
        if gp_vars.ndim == 1 and is_input:
            return gp_vars.reshape(1, -1)
        if gp_vars.ndim in (1, 2):
            return gp_vars
        raise ParameterError("Variables should be an MVar of dimension 1 or 2")
    if isinstance(gp_vars, dict):
        gp_vars = gp_vars.values()
    if isinstance(gp_vars, list):
        if is_input:
            return gp.MVar.fromlist(gp_vars).reshape(1, -1)
        return gp.MVar.fromlist(gp_vars)
    if isinstance(gp_vars, gp.Var):
        return gp.MVar.fromlist([gp_vars]).reshape(1, 1)
    raise ParameterError("Could not validate variables")


class AbstractPredictorConstr(SubModel):
    """Base class to store sub-model added by :py:func:`gurobi_ml.add_predictor_constr`

    This class is the base class to store everything that is added to
    a Gurobi model when embedding a trained predictor into it. Depending on
    the type of the predictor, a class derived from it will be returned
    by :py:func:`gurobi_ml.add_predictor_constr`.

    Warning
    -------

    Users should usually never construct objects of this class and it's inherited
    classes. They are returned by the :py:func:`gurobi_ml.add_predictor_constr` and other
    functions.

    """

    def __init__(self, gp_model, input_vars, output_vars=None, **kwargs):
        self._input = validate_gp_vars(input_vars, True)
        if output_vars is not None:
            self._output = validate_gp_vars(output_vars, False)
        else:
            self._output = None
        SubModel.__init__(self, gp_model, **kwargs)

    def _validate(self):
        """Validate input and output variables (check shapes, reshape if needed)."""
        input_vars = self._input
        output_vars = self._output
        if output_vars.ndim == 1:
            if input_vars.shape[0] == 1:
                output_vars = output_vars.reshape((1, -1))
            else:
                output_vars = output_vars.reshape((-1, 1))

        if output_vars.shape[0] != input_vars.shape[0]:
            raise ParameterError(
                "Non-conforming dimension between "
                + "input variable and output variable: "
                + f"{output_vars.shape[0]} != {input_vars.shape[0]}"
            )

        self._input = input_vars
        self._output = output_vars

    def _build_submodel(self, model, *args, **kwargs):
        """Predict output from input using predictor or transformer"""
        if self._output is None:
            self._create_output_vars(self._input)
        if self._output is not None:
            self._validate()
        else:
            self._input = validate_gp_vars(self._input, True)
        self._mip_model(**kwargs)
        assert self._output is not None
        return self

    def print_stats(self, abbrev=False, file=None):
        """Print statistics on model additions stored by this class

        This function prints detailed statistics on the variables
        and constraints that where added to the model.

        Usually derived classes reimplement this function to provide more
        details about the structure of the additions (type of ML model,
        layers if it's a neural network,...)

        Arguments
        ---------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """

        if abbrev:
            print(
                f"{self._name:13} {self.output.shape.__str__():>14} {len(self.vars):>12} {len(self.constrs):>12} {len(self.qconstrs):>12} {len(self.genconstrs):>12}",
                file=file,
            )
        else:
            super().print_stats(file)
            print(f"Input has shape {self.input.shape}", file=file)
            print(f"Output has shape {self.output.shape}", file=file)

    def _create_output_vars(self, input_vars, name="output"):
        """May be defined in derived class to create the output variables of predictor"""
        try:
            n_outputs = self.n_outputs_
        except AttributeError:
            return
        rval = self._gp_model.addMVar(
            (input_vars.shape[0], n_outputs), lb=-gp.GRB.INFINITY, name=name
        )
        self._gp_model.update()
        self._output = rval

    def _has_solution(self):
        """Returns true if we have a solution"""
        try:
            v = self._input.X
            v = self._output.X
            return True
        except gp.GurobiError:
            pass
        return False

    def get_error(self):
        """Returns error in Gurobi's solution with respect to prediction from input

        Note that this function is implemented in child classes.

        Returns
        -------
        error: ndarray of same shape as :py:attr:`output`
            Assuming that we have a solution for the input and output variables
            `x, y`. Returns the absolute value of the differences between `predict(x)` and
            `y`, where `predict` is the prediction function for the object we are modeling
            (`predict` for Scikit-Learn and Keras, `forward` for Pytorch).

        Raises
        ------
        NoSolution
            If the Gurobi model has no solution (either was not optimized or is infeasible).
        """
        assert False, "Not implemented"

    @staticmethod
    def _indexed_name(index, name):
        index = f"{index}".replace(" ", "")
        return f"{name}[{index}]"

    @property
    def output(self):
        """Returns the output variables of embedded predictor"""
        return self._output

    @property
    def input(self):
        """Returns the input variables of embedded predictor"""
        return self._input

    def __str__(self):
        return self._name
