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
from ._common import (
    TreeLeaves,
    _compute_reachability,
    _goes_left,
    _goes_right,
    _no_reachable_leaf_error,
)
from .biggs_perakis import _biggs_perakis_formulation
from .ensemble_model import ENSEMBLE_FORMULATIONS, add_tree_ensemble_formulation

#: Default value of the ``formulation`` keyword argument, shared by
#: ``AbstractTreeEstimator`` and the ensemble wrapper classes.
DEFAULT_FORMULATION = "leaf"


def _formulation_kwarg(kwargs):
    """The requested ``formulation``, defaulting to :py:data:`DEFAULT_FORMULATION`."""
    return kwargs.get("formulation", DEFAULT_FORMULATION)


class TreeLeavesAccessor:
    """Mixin exposing per-tree leaf variables."""

    _tree_leaves = None

    @property
    def tree_leaves(self):
        """Tuple of :py:class:`TreeLeaves`, one per tree.

        Use it to read which leaf each tree selected, or to add constraints
        on the leaves, for example to require that the solution falls into
        leaves that contain training data.

        Raises
        ------
        AttributeError
            If the formulation does not expose leaf variables (``"paths"``).
        """
        if self._tree_leaves is None:
            raise AttributeError(
                "this predictor constraint's formulation does not expose "
                "per-tree leaf variables"
            )
        return self._tree_leaves


def _flatten_tree_leaves(estimators):
    """Concatenate the per-tree leaves of per-tree ``AbstractTreeEstimator``\\ s.

    Returns ``None`` if any estimator does not expose leaf variables (e.g.
    the ``"paths"`` formulation), matching :py:attr:`TreeLeavesAccessor._tree_leaves`.
    """
    if not all(est._tree_leaves is not None for est in estimators):
        return None
    return tuple(leaves for est in estimators for leaves in est._tree_leaves)


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

    reachability_matrix, node_lb, node_ub, low, high = _compute_reachability(
        gp_model, tree, _input, epsilon, leaf_nodes, safety_floor
    )
    input_ub = _input.getAttr(GRB.Attr.UB)
    input_lb = _input.getAttr(GRB.Attr.LB)

    # Drop leaves that no example can reach — they contribute nothing.
    any_reachable = reachability_matrix.any(axis=0)  # (n_leaves,)
    active_leaf_nodes = leaf_nodes[any_reachable]
    active_reachability = reachability_matrix[:, any_reachable]  # (nex, n_active)
    n_active = len(active_leaf_nodes)

    if n_active == 0:
        raise _no_reachable_leaf_error()

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
            feat_lb = node_lb[feature, node] + high
            feat_ub = node_ub[feature, node] + low
            lb, ub = input_lb[:, feature], input_ub[:, feature]

            # Inputs whose bounds decide the split need no constraint.
            if feat_lb > -GRB.INFINITY:
                tight = (
                    ~_goes_right(lb, node_lb[feature, node], epsilon)
                    & (lb < feat_lb)
                    & reachable
                )
                lhs = leafs_vars[tight, i].tolist()
                rhs = _input[tight, feature].tolist()
                n_indicators += sum(tight)
                for l_var, r_var in zip(lhs, rhs):
                    gp_model.addGenConstrIndicator(
                        l_var, 1, r_var, GRB.GREATER_EQUAL, feat_lb
                    )

            if feat_ub < GRB.INFINITY:
                tight = (
                    ~_goes_left(ub, node_ub[feature, node], epsilon)
                    & (ub > feat_ub)
                    & reachable
                )
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

    return TreeLeaves(leafs_vars, active_leaf_nodes)


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
            fixed_left = _goes_left(node_value, node_threshold, epsilon)
            nodes[fixed_left, right].UB = 0.0
            nodes[~fixed_left, left].UB = 0.0
        else:
            lhs = _input[:, node_feature].tolist()
            rhs = nodes[:, left].tolist()
            gp_model.addConstrs(
                ((rhs[k] == 1) >> (lhs[k] <= node_threshold + min(epsilon, 0.0)))
                for k in range(nex)
            )
            rhs = nodes[:, right].tolist()
            gp_model.addConstrs(
                ((rhs[k] == 1) >> (lhs[k] >= node_threshold + max(epsilon, 0.0)))
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


class AbstractTreeEstimator(TreeLeavesAccessor, AbstractPredictorConstr):
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

        self._formulation = _formulation_kwarg(kwargs)
        if timer is None:
            self._timer = AbstractPredictorConstr._ModelingTimer()
        else:
            self._timer = timer
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def _mip_model(self, **kwargs):
        if self._formulation in ("leafs", "leaf"):
            # "leafs" is a legacy alias of "leaf", kept for backwards
            # compatibility; new code should use "leaf".
            leaves = _leafs_formulation(
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
            self._tree_leaves = (leaves,)
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
        elif self._formulation == "biggs_perakis":
            leaves = _biggs_perakis_formulation(
                self.gp_model,
                self.input,
                self.output,
                self._tree,
                self._epsilon,
                self._name_var,
                self._safety_floor,
            )
            self._tree_leaves = (leaves,)
        elif self._formulation in ENSEMBLE_FORMULATIONS:
            # A lone decision tree is an ensemble of one tree.
            self._tree_leaves = add_tree_ensemble_formulation(
                self.gp_model,
                [self._tree],
                np.ones(1),
                self.input,
                self.output,
                self._formulation,
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
