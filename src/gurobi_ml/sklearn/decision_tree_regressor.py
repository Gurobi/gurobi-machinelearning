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

""" Module for formulating a :external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor`
in a :gurobipy:`model`.
"""

import numpy as np
from gurobipy import GRB

from ..modeling import AbstractPredictorConstr
from .skgetter import SKgetter


def add_decision_tree_regressor_constr(
    gp_model,
    decision_tree_regressor,
    input_vars,
    output_vars=None,
    epsilon=0.0,
    scale=1.0,
    float_type=np.float32,
    **kwargs
):
    """Formulate decision_tree_regressor into gp_model

    The formulation predicts the values of output_vars using input_vars according to decision_tree_regressor.
    See our :ref:`User's Guide <Decision Tree Regression>` for details on the mip formulation used.

    Parameters
    ----------
    gp_model: :gurobipy:`model`
        The gurobipy model where the predictor should be inserted.
    decision_tree_regressor: :external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor`
        The decision tree regressor to insert as predictor.
    input_vars: :gurobipy:`mvar` or :gurobipy:`var` array like
        Decision variables used as input for decision tree in model.
    output_vars: :gurobipy:`mvar` or :gurobipy:`var` array like, optional
        Decision variables used as output for decision tree in model.
    epsilon: float, optional
        Small value used to impose strict inequalities for splitting nodes in MIP formulations.
    scale: float, optional
        Value
    float_type: type, optional
        Float type for the thresholds defining the node splits in the MIP formulation

    Returns
    -------
    DecisionTreeRegressorConstr
        Object containing information about what was added to gp_model to formulate decision_tree_regressor


    Note
    ----
    |VariablesDimensionsWarn|

    Warning
    -------
    Although decision trees with multiple outputs are tested they were never used in
    a non-trivial optimization model. It should be used with care at this point.
    """
    return DecisionTreeRegressorConstr(
        gp_model,
        decision_tree_regressor,
        input_vars,
        output_vars,
        epsilon,
        scale,
        float_type,
        **kwargs
    )


class DecisionTreeRegressorConstr(SKgetter, AbstractPredictorConstr):
    """Class to model trained :external+sklearn:py:class:`sklearn.tree.DecisionTreeRegressor` with gurobipy

    |ClassShort|

    """

    def __init__(
        self,
        gp_model,
        predictor,
        input_vars,
        output_vars=None,
        epsilon=1e-6,
        scale=1.0,
        float_type=np.float32,
        formulation="leafs",
        **kwargs
    ):
        self.epsilon = epsilon
        self.scale = scale
        self.float_type = float_type
        self._default_name = "tree_reg"
        self._formulation = formulation
        SKgetter.__init__(self, predictor, input_vars)
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def _compute_leafs_bounds(self):

        tree = self.predictor.tree_

        node_lb = -np.ones((tree.n_features, tree.capacity)) * GRB.INFINITY
        node_ub = np.ones((tree.n_features, tree.capacity)) * GRB.INFINITY

        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold

        stack = [
            0,
        ]
        while len(stack):
            node = stack.pop()
            left = children_left[node]
            if left < 0:
                continue
            right = children_right[node]
            assert left not in stack
            assert right not in stack
            node_ub[:, right] = node_ub[:, node]
            node_lb[:, right] = node_lb[:, node]
            node_ub[:, left] = node_ub[:, node]
            node_lb[:, left] = node_lb[:, node]

            node_ub[feature[node], left] = threshold[node]
            node_lb[feature[node], right] = threshold[node] + self.epsilon
            stack.append(right)
            stack.append(left)
        return (node_lb, node_ub)

    def _leaf_mip_model(self, **kwargs):
        tree = self.predictor.tree_
        model = self._gp_model

        _input = self._input
        output = self._output
        outdim = output.shape[1]
        nex = _input.shape[0]

        # Collect leafs and non-leafs nodes
        notleafs = tree.children_left >= 0
        leafs = tree.children_left < 0

        leafs_vars = model.addMVar((nex, sum(leafs)), vtype=GRB.BINARY, name="leafs")
        self.nodevars = leafs_vars

        (node_lb, node_ub) = self._compute_leafs_bounds()
        input_ub = _input.UB
        input_lb = _input.LB

        for i, node in enumerate(leafs.nonzero()[0]):
            reachable = (input_ub >= node_lb[:, node]).all(axis=1) & (
                input_lb <= node_ub[:, node]
            ).all(axis=1)
            # Non reachable nodes
            leafs_vars[~reachable, i].UB = 0.0
            # Leaf node:
            lhs = output[reachable, :].tolist()
            rhs = leafs_vars[reachable, i].tolist()
            value = tree.value[node, :, 0]
            model.addConstrs(
                (rhs[k] == 1) >> (lhs[k][i] == value[i])
                for k in range(sum(reachable))
                for i in range(outdim)
            )

            for feature in range(tree.n_features):
                lb = node_lb[feature, node]
                ub = node_ub[feature, node]

                tight = (input_lb[:, feature] < lb) & reachable
                rhs = leafs_vars[tight, i].tolist()
                lhs = _input[tight, feature].tolist()
                model.addConstrs(
                    (rhs[k] == 1) >> (lhs[k] >= lb) for k in range(sum(tight))
                )

                tight = (input_ub[:, feature] > ub) & reachable
                rhs = leafs_vars[tight, i].tolist()
                lhs = _input[tight, feature].tolist()
                model.addConstrs(
                    (rhs[k] == 1) >> (lhs[k] <= ub) for k in range(sum(tight))
                )

        # We should attain 1 leaf
        model.addConstr(leafs_vars.sum(axis=1) == 1)

        output.LB = np.min(tree.value)
        output.UB = np.max(tree.value)

    def _mip_model(self, **kwargs):
        if self._formulation == "leafs":
            return self._leaf_mip_model(**kwargs)
        else:
            return self._paths_mip_model(**kwargs)

    def _paths_mip_model(self, **kwargs):
        tree = self.predictor.tree_
        model = self._gp_model

        _input = self._input
        output = self._output
        outdim = output.shape[1]
        nex = _input.shape[0]
        nodes = model.addMVar((nex, tree.capacity), vtype=GRB.BINARY, name="node")
        self.nodevars = nodes

        # Collect leafs and non-leafs nodes
        notleafs = tree.children_left >= 0
        leafs = tree.children_left < 0

        # Connectivity constraint
        model.addConstr(
            nodes[:, notleafs]
            == nodes[:, tree.children_right[notleafs]]
            + nodes[:, tree.children_left[notleafs]]
        )

        # The value of the root is always 1
        nodes[:, 0].LB = 1.0

        # Node splitting
        for node in notleafs.nonzero()[0]:
            left = tree.children_left[node]
            right = tree.children_right[node]
            threshold = tree.threshold[node]
            threshold = self.float_type(threshold)
            scale = max(abs(1 / threshold), self.scale)
            # Intermediate node
            feature = tree.feature[node]
            feat_var = _input[:, feature]

            fixed_input = (feat_var.UB == feat_var.LB).all()

            if fixed_input:
                # Special case where we have an MVarPlusConst object
                # If that feature is a constant we can directly fix it.
                value = _input[:, feature].LB
                fixed_left = value <= threshold
                nodes[fixed_left, right].UB = 0.0
                nodes[~fixed_left, left].UB = 0.0
            else:
                lhs = _input[:, feature].tolist()
                rhs = nodes[:, left].tolist()
                threshold *= scale
                model.addConstrs(
                    ((rhs[k] == 1) >> (scale * lhs[k] <= threshold)) for k in range(nex)
                )
                rhs = nodes[:, right].tolist()
                model.addConstrs(
                    ((rhs[k] == 1) >> (scale * lhs[k] >= threshold + self.epsilon))
                    for k in range(nex)
                )

        for node in leafs.nonzero()[0]:
            # Leaf node:
            lhs = output.tolist()
            rhs = nodes[:, node].tolist()
            value = tree.value[node, :, 0]
            model.addConstrs(
                (rhs[k] == 1) >> (lhs[k][i] == value[i])
                for k in range(nex)
                for i in range(outdim)
            )

        # We should attain 1 leaf
        model.addConstr(nodes[:, leafs].sum(axis=1) == 1)

        output.LB = np.min(tree.value)
        output.UB = np.max(tree.value)
