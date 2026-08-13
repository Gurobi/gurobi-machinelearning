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

"""Ensemble-level tree formulations built on shared ordered variables.

The formulations dispatched here differ from the per-tree ``"leaf"``,
``"biggs_perakis"`` and ``"paths"`` formulations of
:py:class:`gurobi_ml.modeling.decision_tree.AbstractTreeEstimator` in that
their ordered variables exist once per (feature, threshold) for the whole
ensemble — all trees reference the same variables: the split binaries
``z[f, j]`` ("is ``x_f <= v_j``?") of ``"misic"`` and ``"misic_lazy"``, and
the continuous incremental variables ``mu[f, j]`` of ``"ocean"``. The trees
are therefore coupled through the shared variables instead of only through
the continuous input variables. For ``"misic"``, the binary count grows with
the number of distinct thresholds, not with the number of trees.
"""

from functools import partial
from warnings import warn

import numpy as np

from .misic import SplitVariables, add_misic_tree
from .ocean import IncrementalMuVariables, add_ocean_tree

# Formulation name -> (variables shared by the trees, per-tree builder).
_FORMULATIONS = {
    "misic": (SplitVariables, add_misic_tree),
    # Mišić with its split-linking rows marked lazy (Lazy=3).
    "misic_lazy": (SplitVariables, partial(add_misic_tree, lazy=True)),
    "ocean": (IncrementalMuVariables, add_ocean_tree),
}

#: Formulations handled by :py:func:`add_tree_ensemble_formulation`.
ENSEMBLE_FORMULATIONS = tuple(_FORMULATIONS)


def add_tree_ensemble_formulation(
    gp_model,
    trees,
    weights,
    _input,
    output,
    formulation,
    epsilon,
    _name_var,
    safety_floor=0.0,
    constant=0.0,
    output_coef=1.0,
):
    """Formulate a tree ensemble in gp_model using formulation.

    The formulation predicts
    ``output_coef * output == sum_t weights[t] * tree_t(input) + constant``.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    trees : list of dict
        The trees of the ensemble (dict representation of
        :py:class:`gurobi_ml.modeling.decision_tree.AbstractTreeEstimator`).
    weights : ndarray
        Weight of each tree in the ensemble prediction.
    _input : mvar_array_like
        Decision variables used as input for the ensemble.
    output : mvar_array_like
        Decision variables used as output for the ensemble.
    formulation : str
        The formulation to use; one of ``ENSEMBLE_FORMULATIONS``.
    epsilon : float
        Small value used to impose strict inequalities for splitting nodes.
    _name_var : function
        Function to name variables.
    safety_floor : float, optional
        |SafetyFloorParam|
    constant : float, optional
        Constant offset of the ensemble prediction (e.g. the initial
        prediction of a gradient boosting model). Defaults to 0.
    output_coef : float, optional
        Coefficient of ``output`` in the linking constraint. Averaging
        ensembles pass the number of trees here, so that the leaf values
        enter the model unscaled and the only scaling coefficient is an
        exact integer on ``output``. Defaults to 1.

    Returns
    -------
    tuple of TreeLeaves
        The leaf variables of each tree.
    """
    try:
        shared_variables, tree_builder = _FORMULATIONS[formulation]
    except KeyError:
        raise ValueError(f"Unknown formulation: {formulation}") from None

    # Only the shared binary split variables make epsilon global: they take
    # a value whether or not a tree traverses that split. The epsilon of
    # "ocean" lives in the flow linking, so it acts path-wise like the leaf
    # baseline.
    if epsilon > 0.0 and formulation in ("misic", "misic_lazy"):
        warn(
            f"epsilon={epsilon} with the '{formulation}' formulation applies "
            "globally: the band (t, t + epsilon) of every threshold of the "
            "ensemble is forbidden, not only along the selected paths as in "
            "the 'leaf' formulation. The model becomes infeasible when the "
            "attainable range of an input (e.g. a derived pipeline feature) "
            "is narrower than epsilon, and an epsilon below Gurobi's "
            "IntFeasTol may return solutions that disagree with the "
            "predictor. Consider leaving epsilon at 0.",
            UserWarning,
        )

    shared_vars = shared_variables(
        gp_model, trees, _input, epsilon, _name_var, safety_floor
    )

    outdim = output.shape[1]
    output_lb = np.full(outdim, float(constant))
    output_ub = np.full(outdim, float(constant))
    total = constant
    tree_leaves = []
    for i, (tree, weight) in enumerate(zip(trees, weights)):
        expression, values, leaves = tree_builder(
            gp_model,
            shared_vars,
            tree,
            _input,
            epsilon,
            name=_name_var(f"y[{i}]"),
            safety_floor=safety_floor,
        )
        tree_leaves.append(leaves)
        total = total + weight * expression
        output_lb += np.minimum(
            weight * values.min(axis=0), weight * values.max(axis=0)
        )
        output_ub += np.maximum(
            weight * values.min(axis=0), weight * values.max(axis=0)
        )

    gp_model.addConstr(output_coef * output == total)
    gp_model.addConstr(output_coef * output >= output_lb)
    gp_model.addConstr(output_coef * output <= output_ub)

    return tuple(tree_leaves)
