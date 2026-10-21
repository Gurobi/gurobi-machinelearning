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

"""Mišić tree-ensemble formulation.

Reference: Mišić, "Optimization of Tree Ensembles" (2020),
https://arxiv.org/abs/1705.10883.
"""

import numpy as np
from gurobipy import GRB

from .decision_tree_model import (
    TreeLeaves,
    _compute_reachability,
    _goes_left,
    _goes_right,
    _split_offsets,
)


class SplitVariables:
    """Shared ordered split binaries ``z[f, j]`` of a tree ensemble.

    For every example ``k``, feature ``f`` and distinct split threshold
    ``v_j`` (sorted increasingly, collected over all trees of the ensemble)
    a binary variable models whether ``x[k, f]`` goes left at ``v_j``.  The
    binaries of one feature are ordered by ``z[k, f, j] <= z[k, f, j + 1]``,
    and indicator constraints link them to the input variables, with the
    strict side moved by epsilon (see ``_split_offsets``):

        ``z = 1  =>  x <= v_j + low``    and    ``z = 0  =>  x >= v_j + high``.

    When the bounds of an input variable decide the split, the binary is
    fixed instead, without applying epsilon, like in the leaf formulation.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    trees : list of dict
        The trees of the ensemble (dict representation of
        :py:class:`gurobi_ml.modeling.decision_tree.AbstractTreeEstimator`).
    _input : mvar_array_like
        Decision variables used as input for the ensemble.
    epsilon : float
        Small value used to impose strict inequalities for splitting nodes.
    _name_var : function
        Function to name variables.
    """

    def __init__(self, gp_model, trees, _input, epsilon, _name_var):
        nex, n_features = _input.shape

        input_lb = _input.getAttr(GRB.Attr.LB)
        input_ub = _input.getAttr(GRB.Attr.UB)

        split_features = np.concatenate(
            [tree["feature"][tree["children_left"] >= 0] for tree in trees]
        )
        split_thresholds = np.concatenate(
            [tree["threshold"][tree["children_left"] >= 0] for tree in trees]
        )

        self._thresholds = {}
        self._vars = {}
        for f in range(n_features):
            values = np.unique(split_thresholds[split_features == f])
            if values.size == 0:
                continue

            z = gp_model.addMVar(
                (nex, values.size), vtype=GRB.BINARY, name=_name_var(f"z[{f}]")
            )
            self._thresholds[f] = values
            self._vars[f] = z

            # x <= v_j implies x <= v_{j+1}: the binaries are ordered.
            if values.size > 1:
                gp_model.addConstr(z[:, :-1] <= z[:, 1:])

            # When the input bounds decide a split (including all fixed
            # features), fix the binary; only the remaining ones need linking
            # constraints to the input variables.
            always_left = _goes_left(input_ub[:, [f]], values[None, :], epsilon)
            never_left = _goes_right(input_lb[:, [f]], values[None, :], epsilon)
            z[always_left].setAttr(GRB.Attr.LB, 1.0)
            z[never_left].setAttr(GRB.Attr.UB, 0.0)

            low, high = _split_offsets(epsilon)
            for k, j in zip(*(~(always_left | never_left)).nonzero()):
                z_var = z[k, j].item()
                x_var = _input[k, f].item()
                gp_model.addGenConstrIndicator(
                    z_var, 1, x_var, GRB.LESS_EQUAL, values[j] + low
                )
                gp_model.addGenConstrIndicator(
                    z_var, 0, x_var, GRB.GREATER_EQUAL, values[j] + high
                )

    def column(self, feature, threshold):
        """Return the binaries (one per example) of feature's split at threshold."""
        values = self._thresholds[feature]
        j = np.searchsorted(values, threshold)
        assert j < values.size and values[j] == threshold
        return self._vars[feature][:, j]


def _leaf_intervals(tree):
    """Index the leaves of each subtree by depth-first leaf order.

    In a binary tree the leaves of any subtree are contiguous in left-to-right
    (depth-first) leaf order.  Returns ``(leaves_order, first, last)`` where
    ``leaves_order`` lists the leaf nodes in that order, and the leaves below
    ``node`` are the positions ``first[node]:last[node]`` of ``leaves_order``.
    """
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    first = np.zeros(tree["capacity"], dtype=np.int64)
    last = np.zeros(tree["capacity"], dtype=np.int64)
    leaves_order = []

    stack = [(0, False)]
    while stack:
        node, children_done = stack.pop()
        if children_left[node] < 0:
            first[node] = len(leaves_order)
            leaves_order.append(node)
            last[node] = len(leaves_order)
        elif children_done:
            first[node] = first[children_left[node]]
            last[node] = last[children_right[node]]
        else:
            stack.append((node, True))
            stack.append((children_right[node], False))
            stack.append((children_left[node], False))
    return np.array(leaves_order), first, last


def add_misic_tree(
    gp_model,
    shared_vars,
    tree,
    _input,
    epsilon,
    name=None,
    safety_floor=0.0,
    lazy=False,
):
    """Add the Mišić formulation of one tree of an ensemble to gp_model.

    One continuous variable ``y[k, l] >= 0`` per example and reachable leaf,
    with ``sum_l y[k, l] == 1``; for each splitting node the leaves of its
    left (right) subtree are linked to the ensemble's shared split binary:
    ``sum_{l in left} y <= z`` and ``sum_{l in right} y <= 1 - z``.

    Leaves that no example can reach given the input variable bounds are
    dropped; the pruning uses the same leaf boxes as the leaf formulation.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    shared_vars : SplitVariables
        The shared split binaries of the ensemble.
    tree : dict
        The decision tree to model.
    _input : mvar_array_like
        Decision variables used as input for the ensemble.
    epsilon : float
        Small value used to impose strict inequalities for splitting nodes.
    name : str, optional
        Name for the leaf variables.
    safety_floor : float, optional
        |SafetyFloorParam|
    lazy : bool, optional
        If True, mark the split-linking rows lazy (``Lazy=3``): Gurobi keeps
        them out of the relaxation and adds them when they are violated,
        including at the root.

    Returns
    -------
    tuple
        ``(expression, values)`` where ``expression`` is the tree's output as
        a linear expression of shape ``(n_examples, n_outputs)`` and
        ``values`` are the output values of the reachable leaves.
    """
    nex = _input.shape[0]

    leaves_order, first, last = _leaf_intervals(tree)

    reachability, _, _ = _compute_reachability(
        gp_model,
        tree,
        _input,
        epsilon,
        leaves_order,
        safety_floor,
    )

    # Positions (in depth-first leaf order) of leaves some example can reach.
    active_positions = reachability.any(axis=0).nonzero()[0]
    if active_positions.size == 0:
        raise ValueError(
            "No reachable leaf nodes given the current input variable bounds; "
            "the decision tree constraint would be infeasible."
        )

    y = gp_model.addMVar((nex, active_positions.size), ub=1.0, name=name)
    y[~reachability[:, active_positions]].setAttr(GRB.Attr.UB, 0.0)

    gp_model.addConstr(y.sum(axis=1) == 1)

    children_left = tree["children_left"]
    children_right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]
    for node in (children_left >= 0).nonzero()[0]:
        z_col = shared_vars.column(feature[node], threshold[node])
        for child, is_left in (
            (children_left[node], True),
            (children_right[node], False),
        ):
            begin, end = np.searchsorted(active_positions, (first[child], last[child]))
            if begin == end:
                # No reachable leaf below this child: nothing to link.
                continue
            side = y[:, begin:end].sum(axis=1)
            if is_left:
                split_row = gp_model.addConstr(side <= z_col)
            else:
                split_row = gp_model.addConstr(side <= 1 - z_col)
            if lazy:
                split_row.setAttr(GRB.Attr.Lazy, 3)

    active_leaves = leaves_order[active_positions]
    values = tree["value"][active_leaves, :]
    return y @ values, values, TreeLeaves(y, active_leaves)
