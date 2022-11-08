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

""" Module for embedding a :external+sklearn:py:class:`sklearn.ensemble.RandomForestRegressor`
into a :gurobipy:`model`.
"""

from gurobipy import GRB

from ..modeling import AbstractPredictorConstr
from .decision_tree_regressor import add_decision_tree_regressor_constr
from .skgetter import SKgetter


def add_random_forest_regressor_constr(
    gp_model, random_forest_regressor, input_vars, output_vars=None, **kwargs
):
    """Embed random_forest_regressor into gp_model

    Predict the values of output_vars using input_vars

    Parameters
    ----------
    gp_model: :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    random_forest_regressor: :external+sklearn:py:class:`sklearn.ensemble.RandomForestRegressor`
        The random forest regressor to insert as predictor.
    input_vars: :gurobipy:`mvar` or :gurobipy:`var` array like
        Decision variables used as input for random forest in model.
    output_vars: :gurobipy:`mvar` or :gurobipy:`var` array like, optional
        Decision variables used as output for random forest in model.

    Returns
    -------
    RandomForestRegressorConstr
       Object containing information about what was added to gp_model to embed the
       predictor into it

    Note
    ----
    |VariablesDimensionsWarn|

    Also see :py:func:`gurobi_ml.sklearn.decision_tree_regressor.add_decision_tree_regressor`
    for specific parameters to model decision tree estimators.

    """
    return RandomForestRegressorConstr(
        gp_model, random_forest_regressor, input_vars, output_vars, **kwargs
    )


class RandomForestRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to model trained :external+sklearn:py:class:`sklearn.ensemble.RandomForestRegressor` with gurobipy

    Stores the changes to :gurobipy:`model` when embedding an instance into it."""

    def __init__(self, gp_model, predictor, input_vars, output_vars, **kwargs):
        self.n_outputs_ = predictor.n_outputs_
        self.estimators_ = []
        self._default_name = "rand_forest_reg"
        SKgetter.__init__(self, predictor)
        AbstractPredictorConstr.__init__(self, gp_model, input_vars, output_vars, **kwargs)

    def _mip_model(self, **kwargs):
        """Predict output variables y from input variables X using the
        decision tree.

        Both X and y should be array or list of variables of conforming dimensions.
        """
        model = self._gp_model
        predictor = self.predictor

        _input = self._input
        output = self._output
        nex = _input.shape[0]

        tree_vars = model.addMVar(
            (nex, predictor.n_estimators, self.n_outputs_),
            lb=-GRB.INFINITY,
            name="estimator",
        )

        estimators = []
        for i in range(predictor.n_estimators):
            tree = predictor.estimators_[i]
            estimators.append(
                add_decision_tree_regressor_constr(
                    model, tree, _input, tree_vars[:, i, :], **kwargs
                )
            )
        self.estimators_ = estimators

        model.addConstr(predictor.n_estimators * output == tree_vars.sum(axis=1))

    def print_stats(self, abbrev=False, file=None):
        """Print statistics on model additions stored by this class

        This function prints detailed statistics on the variables
        and constraints that where added to the model.

        Includes a summary of the estimators that it contains.

        Arguments
        ---------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """
        super().print_stats(file=file)
        if abbrev:
            return
        print(file=file)

        header = f"{'Estimator':13} {'Output Shape':>14} {'Variables':>12} {'Constraints':^38}"
        print("-" * len(header), file=file)
        print(header, file=file)
        print(f"{' '*41} {'Linear':>12} {'Quadratic':>12} {'General':>12}", file=file)
        print("=" * len(header), file=file)
        for estimator in self.estimators_:
            estimator.print_stats(abbrev=True, file=file)
            print(file=file)
        print("-" * len(header), file=file)
