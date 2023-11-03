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

"""Module for formulating a
:external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor`
in a :gurobipy:`model`.
"""


from ..modeling.decision_tree import AbstractTreeEstimator
from .skgetter import SKgetter


def add_decision_tree_regressor_constr(
    gp_model,
    decision_tree_regressor,
    input_vars,
    output_vars=None,
    epsilon=0.0,
    **kwargs,
):
    """Formulate decision_tree_regressor into gp_model.

    The formulation predicts the values of output_vars using input_vars
    according to decision_tree_regressor. See our :ref:`User's Guide <Decision
    Tree Regression>` for details on the mip formulation used.

    Parameters
    ----------
    gp_model : :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    decision_tree_regressor : :external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor`
        The decision tree regressor to insert as predictor.
    input_vars : mvar_array_like
        Decision variables used as input for decision tree in model.
    output_vars : mvar_array_like, optional
        Decision variables used as output for decision tree in model.
    epsilon : float, optional
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.
    Returns
    -------
    DecisionTreeRegressorConstr
        Object containing information about what was added to gp_model to
        formulate decision_tree_regressor

    Notes
    -----

    |VariablesDimensionsWarn|

    Warnings
    --------

    Although decision trees with multiple outputs are tested they were never
    used in a non-trivial optimization model. It should be used with care at
    this point.
    """
    return DecisionTreeRegressorConstr(
        gp_model, decision_tree_regressor, input_vars, output_vars, epsilon, **kwargs
    )


class DecisionTreeRegressorConstr(SKgetter, AbstractTreeEstimator):
    """Class to formulate a trained
    :external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor` in a
    gurobipy model.

    |ClassShort|
    """

    def __init__(
        self,
        gp_model,
        predictor,
        input_vars,
        output_vars=None,
        epsilon=0.0,
        formulation="leafs",
        **kwargs,
    ):
        self._default_name = "tree_reg"

        formulations = ("leafs", "paths")
        if formulation not in formulations:
            raise ValueError(
                "Wrong value for formulation should be one of {}.".format(formulations)
            )
        self._formulation = formulation
        SKgetter.__init__(self, predictor, input_vars)
        tree = self.predictor.tree_

        tree_dict = {
            "children_left": tree.children_left,
            "children_right": tree.children_right,
            "feature": tree.feature,
            "threshold": tree.threshold,
            "value": tree.value[:, :, 0],
            "capacity": tree.capacity,
            "n_features": tree.n_features,
        }
        AbstractTreeEstimator.__init__(
            self, gp_model, tree_dict, input_vars, output_vars, epsilon, **kwargs
        )
