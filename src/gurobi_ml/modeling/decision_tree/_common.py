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

"""Helpers shared by the per-tree (``decision_tree_model``) and ensemble
(``ensemble_model``, ``misic``, ``ocean``, ``biggs_perakis``) formulations.

Kept in their own module, rather than in ``decision_tree_model``, so that the
ensemble modules can import them without creating a cycle:
``decision_tree_model`` dispatches to the ensemble formulations at runtime
(``AbstractTreeEstimator._mip_model``), while the ensemble formulations need
these reachability and epsilon helpers at import time.
"""

from typing import NamedTuple
from warnings import warn

import gurobipy as gp
import numpy as np
from gurobipy import GRB


class TreeLeaves(NamedTuple):
    """Leaf variables of one tree.

    ``variables[k, j]`` equals 1 when input row ``k`` reaches the leaf with
    node id ``nodes[j]``, and 0 otherwise. The columns ``j`` run over the
    leaves that some input row can reach given the bounds of the input
    variables; for a row that cannot reach leaf ``j``, the upper bound of
    ``variables[k, j]`` is fixed to 0.

    The kind of variable depends on the formulation: binary variables for
    ``"leaf"`` and ``"biggs_perakis"``, continuous leaf weights for
    ``"misic"`` and ``"misic_lazy"``, continuous leaf flows for ``"ocean"``.
    The continuous variables take the values 0 and 1 in a solution of the
    MIP, but can be fractional in its relaxation (e.g. with
    :external+gurobi:py:meth:`Model.relax`).

    Attributes
    ----------
    variables : :external+gurobi:py:class:`MVar`
        Leaf variables, of shape (number of input rows, number of reachable
        leaves).
    nodes : ndarray
        Node id of each reachable leaf in the tree.
    """

    variables: gp.MVar
    nodes: np.ndarray


def _split_offsets(epsilon):
    """Offsets of the two branches of a split at threshold ``t``.

    The left branch is ``x <= t + low`` and the right branch
    ``x >= t + high``. A positive epsilon makes the right branch strict
    (``x <= t`` goes left, as in scikit-learn and LightGBM); a negative one
    the left branch (``x < t`` goes left, as in XGBoost).
    """
    return min(epsilon, 0.0), max(epsilon, 0.0)


def _goes_left(input_ub, threshold, epsilon):
    """Whether the input bounds alone send every input to the left."""
    if epsilon < 0:
        return input_ub < threshold
    return input_ub <= threshold


def _goes_right(input_lb, threshold, epsilon):
    """Whether the input bounds alone send every input to the right."""
    if epsilon < 0:
        return input_lb >= threshold
    return input_lb > threshold


def _no_reachable_leaf_error():
    """Error raised when no example can reach any leaf given input bounds."""
    return ValueError(
        "No reachable leaf nodes given the current input variable bounds; "
        "the decision tree constraint would be infeasible."
    )


def _collect_split_thresholds(trees, n_features):
    """Distinct split thresholds of a tree ensemble, per feature.

    Returns
    -------
    dict
        Maps a feature index to its sorted array of distinct thresholds,
        over all trees. Features with no split are omitted.
    """
    split_features = np.concatenate(
        [tree["feature"][tree["children_left"] >= 0] for tree in trees]
    )
    split_thresholds = np.concatenate(
        [tree["threshold"][tree["children_left"] >= 0] for tree in trees]
    )
    thresholds = {}
    for f in range(n_features):
        values = np.unique(split_thresholds[split_features == f])
        if values.size:
            thresholds[f] = values
    return thresholds


def _require_finite_split_bounds(input_lb, input_ub, features, formulation):
    """Raise if a feature used in a split has non-finite input bounds.

    ``"ocean"`` and ``"biggs_perakis"`` use the input variable bounds as
    constraint coefficients, so every feature they split on must be bounded.
    """
    for f in features:
        if not (
            np.isfinite(input_lb[:, f]).all() and np.isfinite(input_ub[:, f]).all()
        ):
            raise ValueError(
                f"The {formulation!r} formulation requires finite bounds on "
                f"every input variable used in a split; feature {f} is "
                "unbounded. Bound the input variables (e.g. to the observed "
                "feature ranges) or use another formulation."
            )


def _compute_leafs_bounds(gp_model, tree, safety_floor=0.0):
    """Compute the bounds that define each leaf of the tree

    The bounds are the split thresholds: a node is reached by the inputs
    between ``node_lb`` and ``node_ub``. The formulations move the strict
    side by epsilon (see :py:func:`_split_offsets`), only for the inputs
    whose bounds do not decide the split.

    Parameters
    ----------
    tree : dict
        The decision tree to model.
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
        node_lb[feature[node], right] = node_threshold
        stack.append(right)
        stack.append(left)
    return (node_lb, node_ub)


def _compute_reachability(gp_model, tree, _input, epsilon, nodes, safety_floor=0.0):
    """Which examples can reach which of the given tree nodes.

    A node is reachable for an example when the box of inputs routed to it
    (from :py:func:`_compute_leafs_bounds`) intersects the example's input
    variable bounds. The strict side of the box is moved by epsilon unless
    the example's bounds decide the split, as in the constraints.

    Returns
    -------
    reachability : np.ndarray
        Boolean array of shape ``(n_examples, len(nodes))``.
    node_lb, node_ub : np.ndarray
        Lower and upper bounds of the input box routed to each node of the
        tree, of shape ``(n_features, capacity)``.
    low, high : float
        The split offsets of ``epsilon`` (see :py:func:`_split_offsets`),
        returned so callers that need them do not recompute them.
    """
    (node_lb, node_ub) = _compute_leafs_bounds(gp_model, tree, safety_floor)
    input_lb = _input.getAttr(GRB.Attr.LB)
    input_ub = _input.getAttr(GRB.Attr.UB)

    low, high = _split_offsets(epsilon)
    selected_lb = node_lb[:, nodes]  # (n_features, len(nodes))
    selected_ub = node_ub[:, nodes]
    reachability = np.ones((_input.shape[0], len(nodes)), dtype=bool)
    for f in range(tree["n_features"]):
        lb, ub = input_lb[:, f, None], input_ub[:, f, None]
        node_lb_f, node_ub_f = selected_lb[f, None, :], selected_ub[f, None, :]
        reachability &= _goes_right(lb, node_lb_f, epsilon) | (ub >= node_lb_f + high)
        reachability &= _goes_left(ub, node_ub_f, epsilon) | (lb <= node_ub_f + low)
    return reachability, node_lb, node_ub, low, high
