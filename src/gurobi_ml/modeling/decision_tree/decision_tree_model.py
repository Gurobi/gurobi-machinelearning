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

""" Utilities for modeling decision trees """

from warnings import warn

import numpy as np
from gurobipy import GRB

from ..base_predictor_constr import AbstractPredictorConstr


def _compute_leafs_bounds(tree, epsilon):
    """Compute the bounds that define each leaf of the tree"""
    capacity = tree["capacity"]
    n_features = tree["n_features"]
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]

    node_lb = -np.ones((n_features, capacity)) * GRB.INFINITY
    node_ub = np.ones((n_features, capacity)) * GRB.INFINITY

    stack = [
        0,
    ]

    while len(stack) > 0:
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
        node_lb[feature[node], right] = threshold[node] + epsilon
        stack.append(right)
        stack.append(left)
    return (node_lb, node_ub)


def _leaf_formulation(
    gp_model, _input, output, tree, epsilon, _name_var, verbose, timer
):
    """Formulate decision tree using 'leaf' formulation

    We have one variable per leaf of the tree and a series of indicator to
    define when that leaf is reached.
    """
    nex = _input.shape[0]
    n_features = tree["n_features"]

    # Collect leaf nodes
    leafs = tree["children_left"] < 0
    leafs_vars = gp_model.addMVar(
        (nex, sum(leafs)), vtype=GRB.BINARY, name=_name_var("leafs")
    )

    if verbose:
        timer.timing(f"Added {nex*sum(leafs)} leafs vars")
    (node_lb, node_ub) = _compute_leafs_bounds(tree, epsilon)
    input_ub = _input.getAttr(GRB.Attr.UB)
    input_lb = _input.getAttr(GRB.Attr.LB)

    for i, node in enumerate(leafs.nonzero()[0]):
        reachable = (input_ub >= node_lb[:, node]).all(axis=1) & (
            input_lb <= node_ub[:, node]
        ).all(axis=1)
        # Non reachable nodes
        leafs_vars[~reachable, i].setAttr(GRB.Attr.UB, 0.0)
        # Leaf node:
        rhs = output[reachable, :].tolist()
        lhs = leafs_vars[reachable, i].tolist()
        values = tree["value"][node, :]
        n_indicators = sum(reachable)
        for l_var, r_vars in zip(lhs, rhs):
            for r_var, value in zip(r_vars, values):
                gp_model.addGenConstrIndicator(l_var, 1, r_var, GRB.EQUAL, value)

        for feature in range(n_features):
            feat_lb = node_lb[feature, node]
            feat_ub = node_ub[feature, node]

            if feat_lb > -GRB.INFINITY:
                tight = (input_lb[:, feature] < feat_lb) & reachable
                lhs = leafs_vars[tight, i].tolist()
                rhs = _input[tight, feature].tolist()
                n_indicators += sum(tight)
                for l_var, r_var in zip(lhs, rhs):
                    gp_model.addGenConstrIndicator(
                        l_var, 1, r_var, GRB.GREATER_EQUAL, feat_lb
                    )

            if feat_ub < GRB.INFINITY:
                tight = (input_ub[:, feature] > feat_ub) & reachable
                lhs = leafs_vars[tight, i].tolist()
                rhs = _input[tight, feature].tolist()
                n_indicators += sum(tight)
                for l_var, r_var in zip(lhs, rhs):
                    gp_model.addGenConstrIndicator(
                        l_var, 1, r_var, GRB.LESS_EQUAL, feat_ub
                    )
        if verbose:
            timer.timing(f"Added leaf {node} using {n_indicators} indicators")

    # We should attain 1 leaf
    gp_model.addConstr(leafs_vars.sum(axis=1) == 1)

    if verbose:
        timer.timing(f"Added {nex} linear constraints")

    output.setAttr(GRB.Attr.LB, np.min(tree["value"]))
    output.setAttr(GRB.Attr.UB, np.max(tree["value"]))


def _paths_formulation(gp_model, _input, output, tree, epsilon, _name_var):
    """
       Path formulation for decision tree

    We have one variable for each node of the tree and do a formulation
    that reconsistutes paths through the tree. This is inferior to the
    leaf formulation and is deprecated.
    """

    warn(
        "Path formulation of decision trees is not tested anymore.", DeprecationWarning
    )
    outdim = output.shape[1]
    nex = _input.shape[0]
    nodes = gp_model.addMVar(
        (nex, tree.capacity), vtype=GRB.BINARY, name=_name_var("node")
    )

    children_left = tree["children_left"]
    children_right = tree["children_right"]
    threshold = tree["threshold"]
    feature = tree["feature"]
    value = tree["value"]
    # Collect leafs and non-leafs nodes
    not_leafs = children_left >= 0
    leafs = children_left < 0

    # Connectivity constraint
    gp_model.addConstr(
        nodes[:, not_leafs]
        == nodes[:, children_right[not_leafs]] + nodes[:, children_left[not_leafs]]
    )

    # The value of the root is always 1
    nodes[:, 0].LB = 1.0

    # Node splitting
    for node in not_leafs.nonzero()[0]:
        left = children_left[node]
        right = children_right[node]
        node_threshold = threshold[node]
        # Intermediate node
        node_feature = feature[node]
        feat_var = _input[:, node_feature]

        fixed_input = (feat_var.UB == feat_var.LB).all()

        if fixed_input:
            # Special case where we have an MVarPlusConst object
            # If that feature is a constant we can directly fix it.
            node_value = _input[:, node_feature].LB
            fixed_left = node_value <= node_threshold
            nodes[fixed_left, right].UB = 0.0
            nodes[~fixed_left, left].UB = 0.0
        else:
            lhs = _input[:, feature].tolist()
            rhs = nodes[:, left].tolist()
            gp_model.addConstrs(
                ((rhs[k] == 1) >> (lhs[k] <= node_threshold)) for k in range(nex)
            )
            rhs = nodes[:, right].tolist()
            gp_model.addConstrs(
                ((rhs[k] == 1) >> (lhs[k] >= node_threshold + epsilon))
                for k in range(nex)
            )

    for node in leafs.nonzero()[0]:
        # Leaf node:
        lhs = output.tolist()
        rhs = nodes[:, node].tolist()
        node_value = value[node, :]
        gp_model.addConstrs(
            (rhs[k] == 1) >> (lhs[k][i] == node_value[i])
            for k in range(nex)
            for i in range(outdim)
        )

    output.LB = np.min(tree.value)
    output.UB = np.max(tree.value)


class AbstractTreeEstimator(AbstractPredictorConstr):
    """Abstract class to model a decision tree

    The decision tree should be stored in a dictionary with a similar representation
    as the one that scikit-learn uses:

        "capacity": number of nodes in the tree (size of the arrays that follow),
        "children_left": index of left children (-1 for a leaf)
        "children_right": index of right children (-1 for a leaf)
        "feature": splitting feature of node
        "threshold": threshold for spliting node
        "value": value of the node for output variable
    """

    def __init__(
        self, gp_model, tree, input_vars, output_vars, epsilon, timer=None, **kwargs
    ):
        self._default_name = "tree"
        self._tree = tree
        self._epsilon = epsilon
        if timer is None:
            self._timer = AbstractPredictorConstr._ModelingTimer()
        else:
            self._timer = timer
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def _mip_model(self, **kwargs):
        _leaf_formulation(
            self.gp_model,
            self.input,
            self.output,
            self._tree,
            self._epsilon,
            self._name_var,
            self.verbose,
            self._timer,
        )

    def get_error(self, eps):
        """Functions returns an error for an abstract class

        Child classes should implement this.
        """
        assert False
