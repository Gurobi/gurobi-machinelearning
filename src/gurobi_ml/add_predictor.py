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

"""Define generic function that can add any known trained predictor."""

from .exceptions import NotRegistered
from .modeling.get_convertor import get_convertor
from .registered_predictors import registered_predictors


def add_predictor_constr(gp_model, predictor, input_vars, output_vars=None, **kwargs):
    """Formulate predictor in gp_model.

    The formulation predicts the values of output_vars using input_vars according to
    predictor.

    Parameters
    ----------
    gp_model : :gurobipy:`model`
            The gurobipy model where the predictor should be inserted.
    predictor:
        The predictor to insert.
    input_vars : mvar_array_like
        Decision variables used as input for predictor in gp_model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for predictor in gp_model.

    Returns
    -------
    AbstractPredictorConstr
        Object containing information about what was added to gp_model to insert the
        predictor in it

    Notes
    -----
    The parameters `input_vars` and `output_vars` can be either

     * Gurobipy matrix variables :gurobipy:`mvar`
     * Pandas data frames containing columns of variables or constants
     * Lists of variables
     * Dictionaries of variables

    For internal use in the package they are cast into matrix variables.

    They should have dimensions that conforms with the input/output of the predictor.
    We denote by `n_features` the dimension of the input of the predictor and by
    `n_output` the dimension of the output.

    If they are matrix variables, `input_vars` and `output_vars` can be either of
    shape `(n_features)` and `(n_outputs,)` respectively or `(k, n_features)` and
    `(k, n_outputs)` respectively (with `k >= 1`). The latter form is especially
    useful if the predictor is used to associate different groups of variables
    (e.g. a prediction is made for every time period in a planning horizon).

    If they are pandas dataframe, `input_vars` should have the features as columns and
    `output_vars` the outputs of predictors. Note that the input_vars dataframe may
    have *fixed* columns containing constant values and *variable* columns containing
    gurobipy variables. A column should not mix constants and variables.

    If they are lists or dictionaries, `input_vars` should have length `n_features` and
    `output_vars` should have length `n_output`.

    Rectangular list of lists of variables that can be converted to a matrix shape can also be used.
    """
    convertors = registered_predictors()
    convertor = get_convertor(predictor, convertors)
    if convertor is None:
        raise NotRegistered(type(predictor).__name__)
    return convertor(gp_model, predictor, input_vars, output_vars, **kwargs)
