# Copyright © 2023 Gurobi Optimization, LLC
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

from abc import ABC, abstractmethod

import gurobipy as gp

from ..exceptions import ParameterError
from ._submodel import _SubModel
from ._var_utils import _get_sol_values, validate_input_vars, validate_output_vars


class AbstractPredictorConstr(ABC, _SubModel):
    """Base class to store sub-model added by :py:func:`gurobi_ml.add_predictor_constr`.

    This class is the base class to store everything that is added to
    a Gurobi model when a trained predictor is inserted into it. Depending on
    the type of the predictor, a class derived from this is returned
    by :py:func:`gurobi_ml.add_predictor_constr`.

    Warnings
    --------

    Users shouldn't construct objects of this class or one of its derived
    classes directly. Those objects are returned by the :py:func:`gurobi_ml.add_predictor_constr` and
    other functions.
    """

    def __init__(self, gp_model, input_vars, output_vars=None, **kwargs):
        self._input = input_vars
        self._output = output_vars
        _SubModel.__init__(self, gp_model, **kwargs)

    def _validate(self):
        """Validate input and output variables (check shapes, reshape if needed)."""
        input_vars = self.input
        output_vars = self.output

        if hasattr(self, "_input_shape") and input_vars.shape[1] != self._input_shape:
            raise ParameterError(
                "Input variables dimension doesn't conform with modeling object "
                + f"{type(self)}, input variable dimension: "
                + f"{self._input_shape} != {input_vars.shape[1]}"
            )

        if output_vars.ndim == 1:
            if input_vars.shape[0] == 1:
                output_vars = output_vars.reshape((1, -1))
            else:
                output_vars = output_vars.reshape((-1, 1))

        if (
            hasattr(self, "_output_shape")
            and output_vars.shape[1] != self._output_shape
        ):
            raise ParameterError(
                "Output variables dimension doesn't conform with modeling object "
                + f"{type(self)}, output variable dimension: "
                + f"{output_vars.shape[1]}"
            )

        if output_vars.shape[0] != input_vars.shape[0]:
            raise ParameterError(
                "Non-conforming dimension between "
                + "input variable and output variable: "
                + f"{output_vars.shape[0]} != {input_vars.shape[0]}"
            )

        self._input = input_vars
        self._output = output_vars

    def _build_submodel(self, gp_model, *args, **kwargs):
        """Predict output from input using predictor or transformer."""
        self._input, columns, index = validate_input_vars(self.gp_model, self._input)
        self._input_index = index
        self._input_columns = columns

        if self._output is None:
            self._output = self._create_output_vars(self._input)
        if self._output is not None:
            self._output = validate_output_vars(self._output)
            self._validate()
        self._mip_model(**kwargs)
        assert self._output is not None
        return self

    def _print_container_steps(self, iterations_name, iterable, file):
        header = f"{iterations_name:13} {'Output Shape':>14} {'Variables':>12} {'Constraints':^38}"
        print("-" * len(header), file=file)
        print(header, file=file)
        print(f"{' '*41} {'Linear':>12} {'Quadratic':>12} {'General':>12}", file=file)
        print("=" * len(header), file=file)
        for step in iterable:
            step.print_stats(abbrev=True, file=file)
            print(file=file)
        print("-" * len(header), file=file)

    def print_stats(self, abbrev=False, file=None):
        """Print statistics on model additions stored by this class.

        This function prints detailed statistics on the variables
        and constraints that where added to the model.

        Usually derived classes reimplement this function to provide more
        details about the structure of the additions (type of ML model,
        layers if it's a neural network,...)

        Parameters
        ----------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """

        if abbrev:
            print(
                f"{self._name:13} {self.output.shape.__str__():>14} {len(self.vars):>12} "
                + f"{len(self.constrs):>12} {len(self.qconstrs):>12} {len(self.genconstrs):>12}",
                file=file,
            )
        else:
            super().print_stats(file)
            print(f"Input has shape {self.input.shape}", file=file)
            print(f"Output has shape {self.output.shape}", file=file)

    def _create_output_vars(self, input_vars, name="out"):
        """May be defined in derived class to create the output variables of predictor."""
        try:
            n_outputs = self._output_shape
        except AttributeError:
            return
        output = self.gp_model.addMVar(
            (input_vars.shape[0], n_outputs),
            lb=-gp.GRB.INFINITY,
            name=self._name_var(name),
        )
        self.gp_model.update()
        return output

    def remove(self):
        """Remove from gp_model everything that was added to embed predictor."""
        _SubModel.remove(self)

    @property
    def _has_solution(self):
        """Returns true if we have a solution."""
        try:
            self.input_values
            return True
        except gp.GurobiError:
            pass
        return False

    @abstractmethod
    def get_error(self, eps):
        """Returns error in Gurobi's solution with respect to prediction from input.

        Returns
        -------
        error : ndarray of same shape as :py:attr:`gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.output`
            Assuming that we have a solution for the input and output variables
            `x, y`. Returns the absolute value of the differences between `predictor.predict(x)` and
            `y`. Where predictor is the regression model represented by this object.

        Raises
        ------
        NoSolution
            If the Gurobi model has no solution (either was not optimized or is infeasible).
        """
        ...

    @abstractmethod
    def _mip_model(self, **kwargs):
        """Makes MIP model for the predictor the sub-class implements."""
        ...

    def _indexed_name(self, index, name):
        index = f"{index}".replace(" ", "")
        return self._name_var(f"{name}[{index}]")

    def _name_var(self, name):
        if self._name != "" and not self._no_recording:
            return name
        return None

    @property
    def output(self):
        """Output variables of embedded predictor.

        Returns
        -------
        output : :gurobipy:`MVar`.
        """
        return self._output

    @property
    def input_values(self):
        """Values for the input variables if a solution is known.

        Returns
        -------
        output_value : ndarray or pandas dataframe with values

        Raises
        ------
        NoSolution
            If the Gurobi model has no solution (either was not optimized or is infeasible).
        """

        return _get_sol_values(self.input, self._input_columns, self._input_index)

    @property
    def output_values(self):
        """Values for the output variables if a solution is known.

        Returns
        -------
        output_value : ndarray or pandas dataframe with values

        Raises
        ------
        NoSolution
            If the Gurobi model has no solution (either was not optimized or is infeasible).
        """
        return _get_sol_values(self.output)

    @property
    def input(self):
        """Input variables of embedded predictor.

        Returns
        -------
        output : :gurobipy:`MVar`.
        """
        return self._input

    def __str__(self):
        return self._name
