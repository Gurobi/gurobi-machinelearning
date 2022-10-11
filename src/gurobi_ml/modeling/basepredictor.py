# Copyright © 2022 Gurobi Optimization, LLC
import gurobipy as gp

from ..exceptions import ModelingError
from .submodel import SubModel


def _default_name(predictor):
    """Make a default name for predictor constraint."""
    return type(predictor).__name__.lower()


def validate_gpvars(gpvars, isinput):
    """Put variables into appropriate form (matrix of variable)"""
    if isinstance(gpvars, gp.MVar):
        if gpvars.ndim == 1 and isinput:
            return gpvars.reshape(1, -1)
        if gpvars.ndim in (1, 2):
            return gpvars
        raise ModelingError("Variables should be an MVar of dimension 1 or 2")
    if isinstance(gpvars, dict):
        gpvars = gpvars.values()
    if isinstance(gpvars, list):
        if isinput:
            return gp.MVar.fromlist(gpvars).reshape(1, -1)
        return gp.MVar.fromlist(gpvars)
    if isinstance(gpvars, gp.Var):
        return gp.MVar.fromlist([gpvars]).reshape(1, 1)
    raise ModelingError("Could not validate variables")


class AbstractPredictorConstr(SubModel):
    """Class to define a submodel"""

    def __init__(self, grbmodel, input_vars, output_vars=None, **kwargs):
        self._input = validate_gpvars(input_vars, True)
        if output_vars is not None:
            self._output = validate_gpvars(output_vars, False)
        else:
            self._output = None
        SubModel.__init__(self, grbmodel, **kwargs)

    def _set_output(self, output_vars):
        self._output = validate_gpvars(output_vars, False)

    @staticmethod
    def validate(input_vars, output_vars):
        """Validate input and output variables (check shapes, reshape if needed."""
        if input_vars is None:
            raise ModelingError("No input variables")
        if output_vars is None:
            raise ModelingError("No output variables")
        if output_vars.ndim == 1:
            if input_vars.shape[0] == 1:
                output_vars = output_vars.reshape((1, -1))
            else:
                output_vars = output_vars.reshape((-1, 1))

        if output_vars.shape[0] != input_vars.shape[0]:
            raise ModelingError(
                "Non-conforming dimension between "
                + "input variable and output variable: "
                + f"{output_vars.shape[0]} != {input_vars.shape[0]}"
            )

        return (input_vars, output_vars)

    def _build_submodel(self, model, *args, **kwargs):
        """Predict output from input using predictor or transformer"""
        if self._output is None:
            self._create_output_vars(self._input)
        if self._output is not None:
            self._input, self._output = self.validate(self._input, self._output)
        else:
            self._input = validate_gpvars(self._input, True)
        self.mip_model()
        assert self._output is not None
        return self

    def print_stats(self, file=None):
        super().print_stats(file)
        print(f"Input has shape {self.input.shape}", file=file)
        print(f"Output has shape {self.output.shape}", file=file)

    def mip_model(self):
        """Defined in derived class the mip_model for the predictor"""

    def _create_output_vars(self, input_vars):
        """May be defined in derived class to create the output variables of predictor"""
        try:
            n_outputs = self.n_outputs_
        except AttributeError:
            return
        rval = self._model.addMVar((input_vars.shape[0], n_outputs), lb=-gp.GRB.INFINITY, name="output")
        self._model.update()
        self._output = rval

    def has_solution(self):
        try:
            v = self._input.X
            v = self._output.X
            return True
        except AttributeError:
            print("No solution available")
        return False

    @property
    def output(self):
        return self._output

    @property
    def input(self):
        return self._input

    def __str__(self):
        return self._name
