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

"""Parmentier–Vidal (OCEAN) tree-ensemble formulation.

Reference: Parmentier & Vidal, "Optimal Counterfactual Explanations in
Tree Ensembles" (2021), https://arxiv.org/abs/2106.06631, and its
reference implementation https://github.com/vidalt/OCEAN.

Each tree routes one unit of flow from its root to a leaf, with one
continuous flow variable per node and one binary per depth level steering
the flow left or right. As in the paper, the feature variables ``mu``
shared by the trees are **continuous** — the only binaries are the
per-tree, per-depth branching variables. For each feature, the thresholds
plus the input variable's bounds partition its domain into intervals, and
``mu`` follows the incremental formulation: ``mu[j]`` is the fraction of
interval ``j`` that lies below the input value, ``mu`` is nonincreasing in
``j``, and the input is recovered exactly by
``x == lb + sum_j mu[j] * width[j]`` — no indicator or big-M constraints.
Flow through a split forces the adjacent ``mu`` values: going right forces
the interval below the threshold to be fully crossed, going left forbids
entering the interval above it.

A nonzero ``epsilon`` is absorbed into that partition rather than into
the linking rows: each threshold ``v`` contributes the two boundaries
``v + low`` and ``v + high`` (see ``_split_offsets``), so the interval it
forbids is a hole of the feature's domain with an interval boundary at each
end. Going left binds the lower end, going right the upper one, and both
stay single-term rows. An input whose bounds decide the split needs no
hole. At ``epsilon = 0`` the two boundaries coincide and the encoding is
identical to the plain one.

The interval widths appear as constraint coefficients, so every feature
used in a split must have **finite bounds** (the paper assumes bounded
feature domains); a :py:exc:`ValueError` is raised otherwise.
"""

import numpy as np
from gurobipy import GRB

from ._common import (
    TreeLeaves,
    _collect_split_thresholds,
    _compute_reachability,
    _goes_left,
    _goes_right,
    _no_reachable_leaf_error,
    _require_finite_split_bounds,
    _split_offsets,
)


class IncrementalMuVariables:
    """Continuous incremental variables shared by the whole ensemble."""

    def __init__(self, gp_model, trees, _input, epsilon, _name_var):
        nex, n_features = _input.shape

        input_lb = _input.getAttr(GRB.Attr.LB)
        input_ub = _input.getAttr(GRB.Attr.UB)

        thresholds = _collect_split_thresholds(trees, n_features)
        _require_finite_split_bounds(input_lb, input_ub, thresholds.keys(), "ocean")

        self._epsilon = epsilon
        self._input_lb = input_lb
        self._input_ub = input_ub
        self._thresholds = {}
        self._mu = {}
        self._left_pos = {}
        self._right_pos = {}
        for f, values in thresholds.items():
            # A nonzero epsilon turns the interval at each threshold into a
            # hole of the domain, so both of its ends become boundaries. At
            # epsilon = 0 the two ends coincide and only the thresholds
            # remain.
            low, high = _split_offsets(epsilon)
            left_ends = values + low
            right_ends = values + high
            boundaries = np.unique(np.concatenate([left_ends, right_ends]))
            left_pos = np.searchsorted(boundaries, left_ends)
            right_pos = np.searchsorted(boundaries, right_ends)

            # Interval boundaries per example: bounds plus the boundaries
            # clipped into them. Boundaries outside an example's bounds give
            # zero-width intervals; the corresponding subtrees are pruned by
            # reachability, so their (vacuous) mu values are never forcing.
            clipped = np.clip(
                boundaries[None, :], input_lb[:, [f]], input_ub[:, [f]]
            )  # (nex, m)
            levels = np.concatenate(
                [input_lb[:, [f]], clipped, input_ub[:, [f]]], axis=1
            )
            widths = np.diff(levels, axis=1)  # (nex, m + 1)

            mu = gp_model.addMVar(
                (nex, boundaries.size + 1), ub=1.0, name=_name_var(f"mu[{f}]")
            )
            # x fills the intervals in order: mu is nonincreasing.
            gp_model.addConstr(mu[:, :-1] >= mu[:, 1:])
            # Exact recovery of the input from the interval fractions.
            gp_model.addConstr(
                _input[:, f] == input_lb[:, f] + (mu * widths).sum(axis=1)
            )

            self._thresholds[f] = values
            self._mu[f] = mu
            self._left_pos[f] = left_pos
            self._right_pos[f] = right_pos

    def link_split(self, gp_model, feature, threshold, flow_left, flow_right):
        """Force mu consistent with a split's flow variables.

        Going left requires ``x <= v_j + low``, so the interval starting at
        ``v_j + low`` must not be entered; going right requires
        ``x >= v_j + high``, so the interval ending at ``v_j + high`` must be
        fully crossed. Both ends are boundaries of the partition, so epsilon
        appears only through their indices. An input whose bounds decide the
        split needs no constraint.
        """
        values = self._thresholds[feature]
        j = np.searchsorted(values, threshold)
        if j >= values.size or values[j] != threshold:
            raise ValueError(
                f"No shared incremental variable for feature {feature} at "
                f"threshold {threshold}; the tree was not part of the "
                "ensemble these shared variables were built from."
            )
        mu = self._mu[feature]

        if flow_left is not None:
            near = ~_goes_left(self._input_ub[:, feature], threshold, self._epsilon)
            if near.any():
                gp_model.addConstr(
                    flow_left[near] <= 1 - mu[near, self._left_pos[feature][j] + 1]
                )
        if flow_right is not None:
            near = ~_goes_right(self._input_lb[:, feature], threshold, self._epsilon)
            if near.any():
                gp_model.addConstr(
                    flow_right[near] <= mu[near, self._right_pos[feature][j]]
                )


def add_ocean_tree(
    gp_model, shared_vars, tree, _input, epsilon, name=None, safety_floor=0.0
):
    """Add the OCEAN flow formulation of one tree of an ensemble to gp_model.

    Per-node flows with conservation and one branching binary per depth
    level; the split linking goes through the ensemble's continuous
    incremental ``mu`` variables.

    Returns ``(expression, values, leaves)`` like the other tree builders.
    """
    nex = _input.shape[0]
    children_left = tree["children_left"]
    children_right = tree["children_right"]

    # Breadth-first traversal from the root, recording depths.
    order = [0]
    depth = np.zeros(tree["capacity"], dtype=np.int64)
    for node in order:  # grows while iterating
        left = children_left[node]
        if left >= 0:
            right = children_right[node]
            depth[left] = depth[right] = depth[node] + 1
            order.append(left)
            order.append(right)
    nodes = np.array(order)

    reachability, _, _, _, _ = _compute_reachability(
        gp_model,
        tree,
        _input,
        epsilon,
        nodes,
        safety_floor,
    )

    active = reachability.any(axis=0)
    active_nodes = nodes[active]
    if not any(children_left[node] < 0 for node in active_nodes):
        raise _no_reachable_leaf_error()
    column = {node: i for i, node in enumerate(active_nodes)}

    flow = gp_model.addMVar((nex, active_nodes.size), ub=1.0, name=name)
    flow[~reachability[:, active]].setAttr(GRB.Attr.UB, 0.0)

    gp_model.addConstr(flow[:, 0] == 1)

    split_nodes = [node for node in active_nodes if children_left[node] >= 0]
    if split_nodes:
        n_levels = int(max(depth[node] for node in split_nodes)) + 1
        branch = gp_model.addMVar(
            (nex, n_levels),
            vtype=GRB.BINARY,
            name=None if name is None else f"branch_{name}",
        )

    feature = tree["feature"]
    threshold = tree["threshold"]
    for node in split_nodes:
        level = branch[:, int(depth[node])]
        conservation = 0
        left = children_left[node]
        flow_left = None
        if left in column:
            flow_left = flow[:, column[left]]
            conservation = conservation + flow_left
            gp_model.addConstr(flow_left <= level)
        right = children_right[node]
        flow_right = None
        if right in column:
            flow_right = flow[:, column[right]]
            conservation = conservation + flow_right
            gp_model.addConstr(flow_right <= 1 - level)
        shared_vars.link_split(
            gp_model, feature[node], threshold[node], flow_left, flow_right
        )
        gp_model.addConstr(flow[:, column[node]] == conservation)

    active_leaves = [node for node in active_nodes if children_left[node] < 0]
    leaf_columns = [column[node] for node in active_leaves]
    values = tree["value"][active_leaves, :]
    leaf_flow = flow[:, leaf_columns]
    return leaf_flow @ values, values, TreeLeaves(leaf_flow, np.array(active_leaves))
