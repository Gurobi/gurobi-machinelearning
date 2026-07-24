# Copyright © 2023-2026 Gurobi Optimization, LLC
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

"""Utilities for modeling decision trees"""

from warnings import warn

import numpy as np
from gurobipy import GRB

from ..base_predictor_constr import AbstractPredictorConstr


def _compute_leafs_bounds(gp_model, tree, feature_is_fixed, epsilon, safety_floor=0.0):
    """Compute the bounds that define each leaf of the tree

    Parameters
    ----------
    tree : dict
        The decision tree to model.
    feature_is_fixed : ndarray
        Boolean array indicating if a feature is fixed.
    epsilon : float
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.
    """
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

    feas_tol = gp_model.Params.FeasibilityTol

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

        node_threshold = threshold[node]
        if 0 < abs(node_threshold) < safety_floor:
            node_threshold = np.sign(node_threshold) * safety_floor

        if 0 < abs(node_threshold) < feas_tol:
            warn(
                f"Split threshold {node_threshold} is smaller than Gurobi's "
                f"feasibility tolerance ({feas_tol}). This may lead to numerical issues. "
                "Consider setting 'safety_floor' to a higher value (e.g., 1e-5).",
                UserWarning,
            )

        node_ub[feature[node], left] = node_threshold
        if feature_is_fixed[feature[node]]:
            node_lb[feature[node], right] = node_threshold
        else:
            node_lb[feature[node], right] = node_threshold + epsilon
        stack.append(right)
        stack.append(left)
    return (node_lb, node_ub)


def _leafs_formulation(
    gp_model, _input, output, tree, epsilon, _name_var, verbose, timer, safety_floor=0.0
):
    """Formulate decision tree using 'leafs' formulation

    We have one variable per leaf of the tree and a series of indicator to
    define when that leaf is reached.
    """
    nex = _input.shape[0]
    n_features = tree["n_features"]

    # Collect leaf nodes
    leafs = tree["children_left"] < 0
    leaf_nodes = leafs.nonzero()[0]

    # Get fixed features we don't want to apply the epsilon for them
    feature_is_fixed = (_input.lb == _input.ub).all(axis=0)

    (node_lb, node_ub) = _compute_leafs_bounds(
        gp_model, tree, feature_is_fixed, epsilon, safety_floor
    )
    input_ub = _input.getAttr(GRB.Attr.UB)
    input_lb = _input.getAttr(GRB.Attr.LB)

    # Reachability: compute (nex, n_leaves) without materializing a (nex, n_features, n_leaves) array.
    leaf_lb = node_lb[:, leaf_nodes]  # (n_features, n_leaves)
    leaf_ub = node_ub[:, leaf_nodes]  # (n_features, n_leaves)
    reachability_matrix = np.ones((nex, leaf_nodes.size), dtype=bool)
    for f in range(n_features):
        reachability_matrix &= (input_ub[:, f, None] >= leaf_lb[f, None, :]) & (
            input_lb[:, f, None] <= leaf_ub[f, None, :]
        )

    # Drop leaves that no example can reach — they contribute nothing.
    any_reachable = reachability_matrix.any(axis=0)  # (n_leaves,)
    active_leaf_nodes = leaf_nodes[any_reachable]
    active_reachability = reachability_matrix[:, any_reachable]  # (nex, n_active)
    n_active = len(active_leaf_nodes)

    if n_active == 0:
        raise ValueError(
            "No reachable leaf nodes given the current input variable bounds; "
            "the decision tree constraint would be infeasible."
        )

    leafs_vars = gp_model.addMVar(
        (nex, n_active), vtype=GRB.BINARY, name=_name_var("leafs")
    )

    if verbose:
        timer.timing(f"Added {nex * n_active} leafs vars")

    for i, node in enumerate(active_leaf_nodes):
        reachable = active_reachability[:, i]
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

    # Use only active leaves for bounds — tighter than using all leaves.
    values = tree["value"][active_leaf_nodes, :]
    gp_model.addConstr(output <= np.max(values, axis=0))
    gp_model.addConstr(output >= np.min(values, axis=0))

    if verbose:
        timer.timing(f"Added {nex} linear constraints")


def _paths_formulation(
    gp_model, _input, output, tree, epsilon, _name_var, safety_floor=0.0
):
    """
       Path formulation for decision tree

    We have one variable for each node of the tree and do a formulation
    that reconsistutes paths through the tree. This is inferior to the
    leaf formulation and is deprecated.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    _input : mvar_array_like
        Decision variables used as input for decision tree in gp_model.
    output : mvar_array_like
        Decision variables used as output for decision tree in gp_model.
    tree : dict
        The decision tree to model.
    epsilon : float
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.
    _name_var : function
        Function to name variables.
    """

    warn(
        "Path formulation of decision trees is not tested anymore.", DeprecationWarning
    )
    outdim = output.shape[1]
    nex = _input.shape[0]
    nodes = gp_model.addMVar(
        (nex, tree["capacity"]), vtype=GRB.BINARY, name=_name_var("node")
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

    feas_tol = gp_model.Params.FeasibilityTol

    # Node splitting
    for node in not_leafs.nonzero()[0]:
        left = children_left[node]
        right = children_right[node]
        node_threshold = threshold[node]
        if 0 < abs(node_threshold) < safety_floor:
            node_threshold = np.sign(node_threshold) * safety_floor

        if 0 < abs(node_threshold) < feas_tol:
            warn(
                f"Split threshold {node_threshold} is smaller than Gurobi's "
                f"feasibility tolerance ({feas_tol}). This may lead to numerical issues. "
                "Consider setting 'safety_floor' to a higher value (e.g., 1e-5).",
                UserWarning,
            )

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
            lhs = _input[:, node_feature].tolist()
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

    gp_model.addConstr(output <= np.max(tree["value"], axis=0))
    gp_model.addConstr(output >= np.min(tree["value"], axis=0))


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
        self,
        gp_model,
        tree,
        input_vars,
        output_vars,
        epsilon,
        timer=None,
        safety_floor=0.0,
        **kwargs,
    ):
        self._default_name = "tree"
        self._tree = tree
        self._epsilon = epsilon
        self._safety_floor = safety_floor

        self._formulation = kwargs.get("formulation", "leaf")
        if timer is None:
            self._timer = AbstractPredictorConstr._ModelingTimer()
        else:
            self._timer = timer
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def _mip_model(self, **kwargs):
        if self._formulation in ("leafs", "leaf"):
            _leafs_formulation(
                self.gp_model,
                self.input,
                self.output,
                self._tree,
                self._epsilon,
                self._name_var,
                self.verbose,
                self._timer,
                self._safety_floor,
            )
        elif self._formulation == "paths":
            _paths_formulation(
                self.gp_model,
                self.input,
                self.output,
                self._tree,
                self._epsilon,
                self._name_var,
                self._safety_floor,
            )
        else:
            raise ValueError(f"Unknown formulation: {self._formulation}")

    def get_error(self, eps):
        """Functions returns an error for an abstract class

        Child classes should implement this.
        """
        raise NotImplementedError("Child classes must implement get_error")
