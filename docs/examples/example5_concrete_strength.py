r"""
Concrete Strength Optimization with Tree Ensembles
==================================================

How can we find the inputs that maximize the prediction of a random forest?
We illustrate this with the concrete compressive strength dataset of Yeh
(1998), available from the
`UCI repository <https://doi.org/10.24432/C5PK67>`_.
Its eight features describe ingredient quantities and curing age; the target
is compressive strength in MPa.

We train a small forest, formulate prediction maximization with ``"misic"``,
solve the model, and inspect the optimized features. Only the final step
compares the available tree ensemble formulations.
Every optimization model in this example fits a size-limited Gurobi license.
The small forest is intentional: this is an introduction to surrogate
optimization, not a performance benchmark or a high-accuracy strength model.

.. warning::

   Independent feature bounds do not describe all physically valid mixtures.
   The optimized features are not a validated concrete recipe, and a large
   predicted strength does not establish actual strength. A practical design
   model needs engineering constraints and independent validation.
"""

import time
import warnings

import gurobipy as gp
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor

from gurobi_ml import add_predictor_constr

pd.set_option("display.float_format", "{:.3f}".format)

##############################################################################
# Load the concrete data and train a small forest
# -----------------------------------------------
#
# OpenML downloads the data on the first run and caches it locally. The first
# run therefore requires internet access. Download errors are not suppressed.
# We use explicit feature columns from the dataset, leaving its last column
# as the strength target.

frame = fetch_openml(data_id=4353, as_frame=True).frame
X = frame.iloc[:, :-1].astype(float)
y = frame.iloc[:, -1].astype(float)

forest = RandomForestRegressor(
    n_estimators=5,
    max_depth=3,
    max_features=1 / 3,
    min_samples_leaf=5,
    random_state=0,
).fit(X.to_numpy(), y.to_numpy())

##############################################################################
# Model prediction maximization
# -----------------------------
#
# Each feature is a continuous decision variable bounded by its minimum and
# maximum in the data. The output of the trained forest becomes the objective.
# We maximize this output subject to the feature bounds:
#
# .. math::
#
#    \max_x f(x) \quad \text{subject to }
#    \min_i X_{ij} \le x_j \le \max_i X_{ij} \quad \forall j.
#
# We use ``"misic"``, which shares split decisions across all trees. See
# :ref:`Decision Trees and Tree Ensembles`.
#
# With the default ``epsilon=0``, an optimal solution often sits exactly on a
# split threshold, where the solver may take either branch, so the objective
# can exceed what the forest predicts at the returned input. We therefore use
# a small positive ``epsilon``. It must be larger than Gurobi's ``IntFeasTol``
# (1e-5 by default) to be enforced by the shared formulations. It must also
# survive the single-precision rounding of ``predict``, so it cannot be
# arbitrarily small relative to the features' magnitude; ``2e-5`` is enough
# here, as the ``prediction at x`` entries below confirm. See
# :ref:`Split Thresholds and Exactness`.
#
# This choice is fragile: single precision cannot resolve steps of ``2e-5``
# for features of a few hundred or more (the spacing is about ``6e-5`` near
# 700), so for some features the margin only survives the rounding because of
# where the thresholds fall. A larger value such as ``1e-4`` is safer, at the
# price of excluding a wider interval around every threshold.
#
# The library warns about a nonzero ``epsilon`` with the shared formulations,
# since it forbids a margin around every threshold of the ensemble. We chose
# the value deliberately and silence that warning here.
# The solve function returns only result data before closing its model.

EPSILON = 2e-5


def optimize_strength(formulation="misic"):
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.Threads = 4
        model.Params.TimeLimit = 10
        x = model.addMVar(
            (1, X.shape[1]),
            lb=X.min().to_numpy(),
            ub=X.max().to_numpy(),
            name="features",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="epsilon=", category=UserWarning)
            prediction = add_predictor_constr(
                model, forest, x, formulation=formulation, epsilon=EPSILON
            )
        model.setObjective(prediction.output.sum(), gp.GRB.MAXIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"{formulation}: expected an optimal solution, got {model.Status}"
            )

        solution = pd.DataFrame(x.X, columns=X.columns)
        result = {
            "formulation": formulation,
            "variables": model.NumVars,
            "binaries": model.NumBinVars,
            "linear constraints": model.NumConstrs,
            "indicator constraints": model.NumGenConstrs,
            "status": model.Status,
            "objective [MPa]": model.ObjVal,
            "prediction at x [MPa]": forest.predict(solution.to_numpy())[0],
            "gap": model.MIPGap,
        }
        return result, solution.iloc[0]


##############################################################################
# Solve the concrete model
# ------------------------
#
# Solve once with ``"misic"`` and seed 0. This tiny model should solve to
# optimality; the function raises an explicit error if it does not.

result, solution = optimize_strength()
print(pd.Series(result).to_string())

##############################################################################
# Inspect the optimized features
# ------------------------------
#
# The values below are the ingredient quantities and curing age returned by
# the solver. They are not a validated recipe or a measurement of strength.
#
# Thanks to ``epsilon``, the reported objective and the prediction at the
# returned input should agree up to numerical noise. This is not automatic:
# with ``epsilon=0``, split thresholds are ambiguous, a solver may select a
# different branch than ``predict`` (which also rounds its inputs to single
# precision), and a raw input vector is not an exactness check. See
# :ref:`Split Thresholds and Exactness`.

print(solution.to_string())

##############################################################################
# We can also see where the optimized features sit within the range of the
# data. Each feature is rescaled to [0, 1] between its minimum and maximum.
# The grey bars show the middle half of the data, the blue ticks the medians,
# and the red dots the optimized values.

scaled_lo = (X.quantile(0.25) - X.min()) / (X.max() - X.min())
scaled_hi = (X.quantile(0.75) - X.min()) / (X.max() - X.min())
scaled_med = (X.median() - X.min()) / (X.max() - X.min())
scaled_opt = (solution - X.min()) / (X.max() - X.min())

fig, ax = plt.subplots(figsize=(7, 4))
ypos = range(len(X.columns))
ax.barh(
    ypos,
    (scaled_hi - scaled_lo).to_numpy(),
    left=scaled_lo.to_numpy(),
    color="0.85",
    label="Middle half",
)
ax.scatter(scaled_med, ypos, marker="|", s=200, color="tab:blue", label="Median")
ax.scatter(scaled_opt, ypos, color="tab:red", zorder=3, label="Optimized")
ax.set_yticks(list(ypos), [c.split("(")[0].strip() for c in X.columns])
ax.invert_yaxis()
ax.set_xlim(-0.02, 1.02)
ax.set_xlabel("Feature value (0 = data min, 1 = data max)")
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False)
fig.tight_layout()
plt.show()

##############################################################################
# Finally: compare formulations
# -----------------------------
#
# Once the application and solution are understood, we can solve the same
# model with other formulations. Only the formulation changes: the forest,
# feature bounds, objective, and solver seed stay fixed.
#
# With ``epsilon=0``, some formulations can select incompatible leaves across
# trees, so their objective need not be an achievable forest prediction. The
# positive ``epsilon`` used here removes this ambiguity: every formulation
# should report an objective equal to the forest's prediction at its returned
# input. Different formulations may still return different inputs with the
# same objective because tree predictions are constant within leaves.
#
# This comparison took well under a second locally; expect a few seconds on
# similar hardware, excluding the first dataset download. Allow about one
# minute if solves hit their limits: five solves have a 10-second limit each,
# plus model construction. Progress and total elapsed time are printed below;
# the budget is not a wall-clock guarantee.

FORMULATIONS = ["leaf", "misic", "misic_lazy", "ocean", "biggs_perakis"]

rows = []
started = time.perf_counter()
for run, formulation in enumerate(FORMULATIONS, start=1):
    print(f"[{run}/5] {formulation}: building and solving...", flush=True)
    wall = time.perf_counter()
    result, _ = optimize_strength(formulation)
    result["build + solve [s]"] = time.perf_counter() - wall
    rows.append(result)
    print(
        f"  {result['build + solve [s]']:.2f}s, status {result['status']}, "
        f"objective {result['objective [MPa]']:.4g}, gap {result['gap']:.2%}",
        flush=True,
    )
print(f"Comparison completed in {time.perf_counter() - started:.1f}s.")
comparison = pd.DataFrame(rows).set_index("formulation")
print(comparison.to_string())

##############################################################################
#
# These tiny models are not suitable for ranking solve times. Larger forests,
# repeated seeds, and the surrounding decision constraints can all change the
# relative performance of the formulations. The self-contained
# `concrete and wine notebook
# <https://github.com/Gurobi/gurobi-machinelearning/blob/main/notebooks/tree-ensembles/Concrete%20and%20Wine%20Forest%20Optimization.ipynb>`_
# revisits the concrete and wine applications of Mišić (2020).
# The `drug-design notebook
# <https://github.com/Gurobi/gurobi-machinelearning/blob/main/notebooks/tree-ensembles-merck/Drug%20Design%20with%20Tree%20Ensembles.ipynb>`_
# illustrates application-specific constraints on selected tree leaves.
