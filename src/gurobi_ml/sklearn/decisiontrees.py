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

""" Module for inserting decision tree based regressor into a gurobipy model.
"""

import numpy as np
from gurobipy import GRB

from ..modeling import AbstractPredictorConstr
from .skgetter import SKgetter


def add_decision_tree_regressor_constr(
    grbmodel,
    decision_tree_regressor,
    input_vars,
    output_vars=None,
    epsilon=0.0,
    scale=1.0,
    float_type=np.float32,
    **kwargs
):
    """Use a `decision_tree_regressor` to predict the value of `output_vars` using `input_vars` in `grbmodel`

    Parameters
    ----------
    grbmodel: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
        The gurobipy model where the predictor should be inserted.
    decision_tree_regressor: :external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor`
        The decision tree regressor to insert as predictor.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.
    epsilon: float, optional
        Small value used to impose strict inequalities for splitting nodes in MIP formulations.
    scale: float, optional
        Value
    float_type: type, optional
        Float type for the thresholds defining the node splits in the MIP formulation

    Returns
    -------
    DecisionTreeRegressorConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars

    Warning
    -------
    Although the formulation of the decision tree with multiple output is regularly tested and should
    be correct, it was never used in a non-trivial optimization model. It should be used with care at
    this point.
    """
    return DecisionTreeRegressorConstr(
        grbmodel,
        decision_tree_regressor,
        input_vars,
        output_vars,
        epsilon,
        scale,
        float_type,
        **kwargs
    )


def add_gradient_boosting_regressor_constr(
    grbmodel, gradient_boosting_regressor, input_vars, output_vars=None, **kwargs
):
    """Use a `gradient_boosting_regressor` to predict the value of `output_vars` using `input_vars` in `grbmodel`

    Parameters
    ----------
    grbmodel: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
              The gurobipy model where the predictor should be inserted.
    gradient_boosting_regressor: :external+sklearn:py:class:`sklearn.ensemble.GradientBoostingRegressor`
        The gradient boosting regressor to insert as predictor.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    GradientBoostingRegressorConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars
    """
    return GradientBoostingRegressorConstr(
        grbmodel, gradient_boosting_regressor, input_vars, output_vars, **kwargs
    )


def add_random_forest_regressor_constr(
    grbmodel, random_forest_regressor, input_vars, output_vars=None, **kwargs
):
    """Use a `random_forest_regressor` to predict the value of `output_vars` using `input_vars` in `grbmodel`

    Parameters
    ----------
    grbmodel: `gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>`_
              The gurobipy model where the predictor should be inserted.
    random_forest_regressor: :external+sklearn:py:class:`sklearn.ensemble.RandomForestRegressor`
        The random forest regressor to insert as predictor.
    input_vars: mvar_array_like
        Decision variables used as input for predictor in model.
    output_vars: mvar_array_like, optional
        Decision variables used as output for predictor in model.

    Returns
    -------
    RandomForestRegressorConstr
        Object containing information about what was added to model to insert the
        predictor in it

    Note
    ----
    See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars

    """
    return RandomForestRegressorConstr(
        grbmodel, random_forest_regressor, input_vars, output_vars, **kwargs
    )


class DecisionTreeRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to model a trained decision tree in a Gurobi model"""

    def __init__(
        self,
        grbmodel,
        predictor,
        input_vars,
        output_vars=None,
        epsilon=0.0,
        scale=1.0,
        float_type=np.float32,
        **kwargs
    ):
        self.n_outputs_ = predictor.n_outputs_
        self.epsilon = epsilon
        self.scale = scale
        self.float_type = float_type
        SKgetter.__init__(self, predictor)
        AbstractPredictorConstr.__init__(self, grbmodel, input_vars, output_vars, **kwargs)

    def _mip_model(self):
        tree = self.predictor.tree_
        model = self._model

        _input = self._input
        output = self._output
        outdim = output.shape[1]
        assert outdim == self.n_outputs_
        nex = _input.shape[0]
        nodes = model.addMVar((nex, tree.capacity), vtype=GRB.BINARY, name="node")
        self.nodevars = nodes

        # Intermediate nodes constraints
        # Can be added all at once
        notleafs = tree.children_left >= 0
        leafs = tree.children_left < 0
        model.addConstr(nodes[:, notleafs] >= nodes[:, tree.children_left[notleafs]])
        model.addConstr(nodes[:, notleafs] >= nodes[:, tree.children_right[notleafs]])
        model.addConstr(
            nodes[:, notleafs]
            <= nodes[:, tree.children_right[notleafs]] + nodes[:, tree.children_left[notleafs]]
        )
        model.addConstr(
            nodes[:, tree.children_right[notleafs]] + nodes[:, tree.children_left[notleafs]] <= 1
        )

        # Node splitting
        for node in range(tree.capacity):
            left = tree.children_left[node]
            right = tree.children_right[node]
            threshold = tree.threshold[node]
            threshold = self.float_type(threshold)
            scale = max(abs(1 / threshold), self.scale)
            if left >= 0:
                # Intermediate node
                model.addConstrs(
                    (nodes[k, left].item() == 1)
                    >> (scale * _input[k, tree.feature[node]] <= scale * threshold)
                    for k in range(nex)
                )
                model.addConstrs(
                    (nodes[k, right].item() == 1)
                    >> (scale * _input[k, tree.feature[node]] >= scale * threshold + self.epsilon)
                    for k in range(nex)
                )
            else:
                # Leaf node:
                model.addConstrs(
                    (nodes[k, node].item() == 1) >> (output[k, i] == tree.value[node][i][0])
                    for i in range(self.n_outputs_)
                    for k in range(nex)
                )

        # We should attain 1 leaf
        model.addConstr(nodes[:, leafs].sum(axis=1) == 1)

        output.LB = np.min(tree.value)
        output.UB = np.max(tree.value)


class GradientBoostingRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to model a trained gradient boosting tree in a Gurobi model"""

    def __init__(self, grbmodel, predictor, input_vars, output_vars, **kwargs):
        self.n_outputs_ = 1
        self.estimators_ = []
        SKgetter.__init__(self, predictor)
        AbstractPredictorConstr.__init__(self, grbmodel, input_vars, output_vars, **kwargs)

    def _mip_model(self):
        """Predict output variables y from input variables X using the
        decision tree.

        Both X and y should be array or list of variables of conforming dimensions.
        """
        model = self._model
        predictor = self.predictor

        _input = self._input
        output = self._output
        nex = _input.shape[0]

        outdim = output.shape[1]
        assert outdim == 1, "Output dimension of gradient boosting regressor should be 1"

        treevars = model.addMVar(
            (nex, predictor.n_estimators_, 1), lb=-GRB.INFINITY, name="estimator"
        )
        constant = predictor.init_.constant_

        estimators = []
        for i in range(predictor.n_estimators_):
            tree = predictor.estimators_[i]
            estimators.append(
                DecisionTreeRegressorConstr(
                    model, tree[0], _input, treevars[:, i, :], default_name="gbt_tree"
                )
            )
        self.estimators_ = estimators

        model.addConstr(output == predictor.learning_rate * treevars.sum(axis=1) + constant[0][0])


class RandomForestRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to model a trained random forest regressor in a Gurobi model"""

    def __init__(self, grbmodel, predictor, input_vars, output_vars, **kwargs):
        self.n_outputs_ = predictor.n_outputs_
        self.estimators_ = []
        SKgetter.__init__(self, predictor)
        AbstractPredictorConstr.__init__(self, grbmodel, input_vars, output_vars, **kwargs)

    def _mip_model(self):
        """Predict output variables y from input variables X using the
        decision tree.

        Both X and y should be array or list of variables of conforming dimensions.
        """
        model = self._model
        predictor = self.predictor

        _input = self._input
        output = self._output
        nex = _input.shape[0]

        treevars = model.addMVar(
            (nex, predictor.n_estimators, self.n_outputs_),
            lb=-GRB.INFINITY,
            name="estimator",
        )

        estimators = []
        for i in range(predictor.n_estimators):
            tree = predictor.estimators_[i]
            estimators.append(
                DecisionTreeRegressorConstr(
                    model, tree, _input, treevars[:, i, :], default_name="rf_tree"
                )
            )
        self.estimators_ = estimators

        model.addConstr(predictor.n_estimators * output == treevars.sum(axis=1))
