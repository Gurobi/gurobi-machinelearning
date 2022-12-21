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

from abc import ABC, abstractmethod

import gurobipy as gp
import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

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


class AbstractPredictorConstr(ABC, SubModel):
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
        self._input = input_vars
        self._output = output_vars
        SubModel.__init__(self, gp_model, **kwargs)

    def to_mvar(self, df, is_input: bool):
        """Function to convert the dataframe into an mlinexpr"""
        if is_input:
            if isinstance(df, pd.DataFrame):
                data = df.to_numpy()
                columns = df.columns
                index = df.index
            elif isinstance(df, pd.Series):
                data = df.to_numpy()
                columns = df.columns
                raise NotImplemented("Input variable as pd.Series is not implemented")
            else:
                data = df

                if len(df.shape) == 2:
                    main_shape = df.shape[1]
                    minor_shape = df.shape[0]
                else:
                    main_shape = df.shape[0]
                    minor_shape = 1
                columns = [f"feat{i}" for i in range(main_shape)]
                index = list(range(minor_shape))
            # If variable is an input we can convert it to an MLinexp
            # but it's better to have an MVar so try this first
            if all(map(lambda i: isinstance(i, gp.Var), data.ravel())):
                rval = gp.MVar.fromlist(data.tolist())
                if len(rval.shape) == 1:
                    rval = rval.reshape(1, -1)
                return rval

            model = self._gp_model
            rval = np.zeros(data.shape, dtype=object)
            const_indices = []
            for i, a in enumerate(data.T):
                if all(map(lambda i: isinstance(i, gp.Var), a)):
                    rval[:, i] = a
                    continue
                try:
                    rval[:, i] = a.astype(np.float64)
                except TypeError:
                    raise TypeError(
                        "Dataframe can't be converted to a linear expression"
                    )
                const_indices.append(i)

            mvar = model.addMVar((data.shape[0], len(const_indices)))
            for i, j in enumerate(const_indices):
                mvar[:, i].LB = rval[:, j]
                mvar[:, i].UB = rval[:, j]
                mvar[:, i].VarName = [f"{columns[j]}[{k}]" for k in index]
                rval[:, j] = mvar[:, i].tolist()
            model.update()
            rval = gp.MVar.fromlist(rval)
            return rval

        if isinstance(df, (pd.DataFrame, pd.Series)):
            data = df.to_numpy()
        else:
            data = df
        if any(map(lambda i: not isinstance(i, gp.Var), data.ravel())):
            raise TypeError("Dataframe can't be converted to an MVar")
        rval = gp.MVar.fromlist(df.tolist())
        if len(rval.shape) == 1:
            rval = rval.reshape(-1, 1)
        return rval

    def validate_gp_vars(self, gp_vars: gp.MVar, is_input: bool):
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
        if HAS_PANDAS:
            if isinstance(gp_vars, (pd.DataFrame, pd.Series, np.ndarray)):
                gp_vars = self.to_mvar(gp_vars, is_input)
                return gp_vars
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

    def _build_submodel(self, gp_model, *args, **kwargs):
        """Predict output from input using predictor or transformer"""
        self._input = self.validate_gp_vars(self._input, True)
        if self._output is None:
            self._create_output_vars(self._input)
        if self._output is not None:
            self._output = self.validate_gp_vars(self._output, False)
            self._validate()
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
                f"{self._name:13} {self.output.shape.__str__():>14} {len(self.vars):>12} "
                + f"{len(self.constrs):>12} {len(self.qconstrs):>12} {len(self.genconstrs):>12}",
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

    @property
    def _has_solution(self):
        """Returns true if we have a solution"""
        try:
            self.input_values
            self._output.X
            return True
        except gp.GurobiError:
            pass
        return False

    @abstractmethod
    def get_error(self):
        """Returns error in Gurobi's solution with respect to prediction from input

        Returns
        -------
        error: ndarray of same shape as :py:attr:`gurobi_ml.modeling.base_predictor_constr.AbstractPredictorConstr.output`
            Assuming that we have a solution for the input and output variables
            `x, y`. Returns the absolute value of the differences between `predictor.predict(x)` and
            `y`. Where predictor is the Pytorch model this object is modeling.
        Raises
        ------
        NoSolution
            If the Gurobi model has no solution (either was not optimized or is infeasible).
        """
        ...

    @abstractmethod
    def _mip_model(self, **kwargs):
        """Makes MIP model for the predictor the sub-class implements"""
        ...

    @staticmethod
    def _indexed_name(index, name):
        index = f"{index}".replace(" ", "")
        return f"{name}[{index}]"

    @property
    def output(self):
        """Returns the output variables of embedded predictor"""
        return self._output

    @property
    def input_values(self):
        if isinstance(self._input, (gp.MLinExpr,)):
            return self.input.getValue()
        else:
            return self.input.X

    @property
    def input(self):
        """Returns the input variables of embedded predictor"""
        return self._input

    def __str__(self):
        return self._name
