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

"""Per-tree constraints of the Biggs–Perakis projected formulation.

Reference: Biggs & Perakis, "Tight Mixed-Integer Optimization Formulations
for Prescriptive Trees", Machine Learning 114 (2025),
https://arxiv.org/abs/2302.14744 — the
"projected union of polyhedra" formulation, the best performer in the
paper's experiments.

Per tree: one binary leaf variable per reachable leaf with
``sum(delta) == 1`` and, for every feature the tree splits on, the two
aggregated box constraints

    sum_l ub[l, i] * delta[l]  >=  x[i]  >=  sum_l lb[l, i] * delta[l]

where ``lb``/``ub`` are the leaf boxes (with the same epsilon and
``feature_is_fixed`` semantics as the leaf formulation) clipped to the
input variable bounds. When ``delta`` selects one leaf the input lies in that
leaf's box; the LP relaxation, projected onto ``x``, is the convex hull of
the union of the leaf boxes — locally ideal per tree. Trees are coupled
through the input variables only (no shared split variables), so epsilon
acts path-wise as in the leaf baseline and, at a threshold, trees may take
different branches.

The box bounds appear as constraint coefficients, so every feature used in
a split must have **finite bounds**; a :py:exc:`ValueError` is raised
otherwise (same requirement and rationale as the "ocean" formulation).
"""

import numpy as np
from gurobipy import GRB

from .decision_tree_model import (
    TreeLeaves,
    _compute_reachability,
)


def _biggs_perakis_formulation(
    gp_model, _input, output, tree, epsilon, _name_var, safety_floor=0.0
):
    """Formulate decision tree using the Biggs–Perakis projected formulation

    The formulation shares no variables between trees, so each tree of an
    ensemble is formulated on its own, like the "leaf" formulation.

    Returns the :py:class:`TreeLeaves` of the tree.
    """
    nex = _input.shape[0]

    input_lb = _input.getAttr(GRB.Attr.LB)
    input_ub = _input.getAttr(GRB.Attr.UB)
    feature_is_fixed = (input_lb == input_ub).all(axis=0)

    split_features = np.unique(tree["feature"][tree["children_left"] >= 0])
    for f in split_features:
        if not (
            np.isfinite(input_lb[:, f]).all() and np.isfinite(input_ub[:, f]).all()
        ):
            raise ValueError(
                "The 'biggs_perakis' formulation requires finite bounds on "
                f"every input variable used in a split; feature {f} is "
                "unbounded. Bound the input variables (e.g. to the observed "
                "feature ranges) or use another formulation."
            )

    # Collect leaf nodes
    leafs = tree["children_left"] < 0
    leaf_nodes = leafs.nonzero()[0]

    reachability_matrix, node_lb, node_ub = _compute_reachability(
        gp_model, tree, _input, feature_is_fixed, epsilon, leaf_nodes, safety_floor
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
    leafs_vars[~active_reachability].setAttr(GRB.Attr.UB, 0.0)

    gp_model.addConstr(leafs_vars.sum(axis=1) == 1)

    for f in split_features:
        # Leaf boxes clipped to each example's input bounds: coefficients
        # stay finite and the selected leaf's box never cuts the example's
        # own domain.
        box_lb = np.clip(
            node_lb[f, active_leaf_nodes][None, :],
            input_lb[:, [f]],
            input_ub[:, [f]],
        )
        box_ub = np.clip(
            node_ub[f, active_leaf_nodes][None, :],
            input_lb[:, [f]],
            input_ub[:, [f]],
        )
        gp_model.addConstr((leafs_vars * box_lb).sum(axis=1) <= _input[:, f])
        gp_model.addConstr((leafs_vars * box_ub).sum(axis=1) >= _input[:, f])

    # Use only active leaves for bounds — tighter than using all leaves.
    values = tree["value"][active_leaf_nodes, :]
    gp_model.addConstr(output == leafs_vars @ values)
    gp_model.addConstr(output <= np.max(values, axis=0))
    gp_model.addConstr(output >= np.min(values, axis=0))

    return TreeLeaves(leafs_vars, active_leaf_nodes)
