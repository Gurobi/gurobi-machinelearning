"""Tests for the tree ensemble formulations ``"misic"``, ``"misic_lazy"``,
``"ocean"`` and ``"biggs_perakis"``.

Beyond the fixed-input equivalence tests in test_sklearn_formulations.py,
this module checks:

- cross-formulation agreement: the misic optimum is always reproduced by
  ``predict`` on one of the threshold cells the solution can represent; the
  leaf optimum dominates it and equals it whenever the leaf solution does not
  tie a threshold duplicated across trees (at such a tie the leaf
  formulation may mix branch choices across trees — a combination no real
  input reproduces — while the shared ``z`` forbids the mix by construction);
  ocean and biggs_perakis lie between misic and leaf;
- existing-coverage parity for ``epsilon`` and fixed features, and the
  warning of misic for a positive epsilon;
- lifecycle: ``remove()`` restores the model, an unknown formulation raises,
  ocean and biggs_perakis require finite bounds;
- model size: with unbounded inputs the misic binary count equals the number
  of distinct split thresholds — independent of the number of trees;
- ``print_stats`` output, and the lazy rows of misic_lazy.
"""

import io
import unittest
import warnings
from types import SimpleNamespace

import gurobipy as gp
import numpy as np
from gurobipy import GRB, GurobiError
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from gurobi_ml import add_predictor_constr

from ..tree_ensemble import (
    representable_cells,
    thresholds_by_feature,
    ties_duplicated_threshold,
)

# Cross-formulation comparisons must run at the default epsilon=0: a positive
# epsilon has a different scope per formulation — path-wise in "leaf" but
# ensemble-wise (every threshold, all trees) in misic —
# so the feasible sets, and hence the optimal objectives, may legitimately
# differ. Positive epsilons are additionally treacherous below Gurobi's
# IntFeasTol (default 1e-5): the indicator linking is translated into SOS1
# constraints whose feasibility check treats values up to IntFeasTol as zero,
# so with epsilon <= IntFeasTol the solver may return an input sitting on the
# wrong side of a threshold by exactly epsilon (observed with epsilon=1e-5:
# MaxVio == epsilon, vanishing for epsilon=1e-4 or IntFeasTol=1e-9).
EPSILON = 0.0


def _sklearn_trees(predictor):
    """Return the fitted ``Tree`` objects of a sklearn tree predictor."""
    if isinstance(predictor, GradientBoostingRegressor):
        return [est[0].tree_ for est in predictor.estimators_]
    if isinstance(predictor, RandomForestRegressor):
        return [est.tree_ for est in predictor.estimators_]
    return [predictor.tree_]


def _tree_pairs(predictor):
    """The (feature, threshold) pairs of each tree of predictor.

    sklearn's ``predict`` casts inputs to float32 before comparing them to
    the (float64) thresholds, hence the float32-aware helpers in
    ``tests.tree_ensemble``.
    """
    pairs = []
    for tree in _sklearn_trees(predictor):
        split_nodes = tree.children_left >= 0
        pairs.append(set(zip(tree.feature[split_nodes], tree.threshold[split_nodes])))
    return pairs


class TestCrossFormulationAgreement(unittest.TestCase):
    """Compare the optima of the formulations on small diabetes models.

    ``setUpClass`` solves every formulation once per case; the tests only
    compare the solutions.
    """

    FORMULATIONS = ("leaf", "misic", "ocean", "biggs_perakis")

    @classmethod
    def setUpClass(cls):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        cls.cases = []
        for random_state in (0, 1):
            predictors = [
                GradientBoostingRegressor(
                    n_estimators=5, max_depth=3, random_state=random_state
                ),
                RandomForestRegressor(
                    n_estimators=5, max_depth=3, random_state=random_state
                ),
                DecisionTreeRegressor(max_depth=4, random_state=random_state),
            ]
            for predictor in predictors:
                predictor.fit(X, y)
                # Per tree: for the duplicated-threshold ties. Per feature:
                # for the cells a solution can represent.
                tree_pairs = _tree_pairs(predictor)
                thresholds = thresholds_by_feature(tree_pairs)
                for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
                    solutions = {
                        formulation: cls._optimize(predictor, X, formulation, sense)
                        for formulation in cls.FORMULATIONS
                    }
                    cls.cases.append(
                        SimpleNamespace(
                            random_state=random_state,
                            predictor=predictor,
                            sense=sense,
                            solutions=solutions,
                            tree_pairs=tree_pairs,
                            thresholds=thresholds,
                        )
                    )

    @staticmethod
    def _optimize(predictor, X, formulation, sense):
        """Solve max/min of the prediction over the box of feature ranges.

        Returns ``(status, max_vio, feas_tol, objective, x_star)``.
        """
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, X.shape[1]), lb=X.min(axis=0), ub=X.max(axis=0))
            pred_constr = add_predictor_constr(
                gpm, predictor, x, epsilon=EPSILON, formulation=formulation
            )
            gpm.setObjective(pred_constr.output.sum(), sense)
            try:
                gpm.optimize()
            except GurobiError as error:
                if error.errno == 10010:
                    warnings.warn(UserWarning("Limited license"))
                    raise unittest.SkipTest("Model too large for limited license")
                raise
            return (
                gpm.Status,
                gpm.MaxVio,
                gpm.Params.FeasibilityTol,
                gpm.ObjVal,
                np.array(x.X[0, :]),
            )

    def _solution(self, case, formulation):
        """Check that formulation was solved and return
        ``(objective, x_star, feas_tol)``."""
        status, max_vio, feas_tol, objective, x_star = case.solutions[formulation]
        self.assertEqual(status, GRB.OPTIMAL, msg=formulation)
        # At epsilon=0 there is no band for a solution to hide in; any
        # violation above FeasibilityTol means a solver tolerance is masking
        # a wrong-side threshold again (see the EPSILON note).
        self.assertLessEqual(max_vio, feas_tol, msg=formulation)
        return objective, x_star, feas_tol

    def _assert_predict_reproduces(self, case, objective, x_star, feas_tol):
        """The MIP optimum must be attained by ``predict`` on one of the
        threshold cells the solution can represent."""
        cells = representable_cells(x_star, case.thresholds, feas_tol)
        predictions = case.predictor.predict(cells)
        tolerance = 1e-4 * (1.0 + abs(objective))
        self.assertLessEqual(
            np.min(np.abs(predictions - objective)),
            tolerance,
            msg="MIP optimum not reproduced by predict on any representable cell",
        )

    def test_diabetes_agreement_misic(self):
        """The misic optimum is reproduced by predict and bounded by the leaf
        optimum; without a duplicated-threshold tie the two agree."""
        for case in self.cases:
            with self.subTest(
                predictor=type(case.predictor).__name__,
                random_state=case.random_state,
                sense=case.sense,
            ):
                obj_leaf, x_leaf, feas_tol = self._solution(case, "leaf")
                objective, x_star, _ = self._solution(case, "misic")
                # The shared z couples the trees, so a misic optimum always
                # corresponds to a real input.
                self._assert_predict_reproduces(case, objective, x_star, feas_tol)

                # Each solve is proven optimal within the default relative
                # MIPGap of 1e-4.
                tolerance = 3e-4 * max(1.0, abs(objective))

                # Every misic-feasible point is leaf-feasible, so the leaf
                # optimum dominates.
                if case.sense == GRB.MAXIMIZE:
                    self.assertGreaterEqual(obj_leaf, objective - tolerance)
                else:
                    self.assertLessEqual(obj_leaf, objective + tolerance)

                # Without a duplicated-threshold tie the leaf formulation
                # cannot mix branches across trees: the optima must agree and
                # the leaf optimum must also correspond to a real input.
                if not ties_duplicated_threshold(case.tree_pairs, x_leaf, feas_tol):
                    self.assertLessEqual(abs(obj_leaf - objective), tolerance)
                    self._assert_predict_reproduces(case, obj_leaf, x_leaf, feas_tol)

    def _check_between_misic_and_leaf(self, formulation):
        """Sandwich misic <= formulation <= leaf (maximize; reversed when
        minimizing).

        Every misic solution is feasible for formulation, and every
        formulation solution for leaf. At a threshold, ocean, biggs_perakis
        and leaf all let each tree pick any leaf whose closed box contains x,
        so in exact arithmetic they equal leaf. Only misic always gives a
        point that predict reproduces, so the predict check skips
        duplicated-threshold ties."""
        for case in self.cases:
            with self.subTest(
                predictor=type(case.predictor).__name__,
                random_state=case.random_state,
                sense=case.sense,
            ):
                obj_leaf, _, feas_tol = self._solution(case, "leaf")
                obj_misic, _, _ = self._solution(case, "misic")
                objective, x_star, _ = self._solution(case, formulation)
                tolerance = 3e-4 * max(1.0, abs(objective))
                if case.sense == GRB.MAXIMIZE:
                    self.assertGreaterEqual(objective, obj_misic - tolerance)
                    self.assertLessEqual(objective, obj_leaf + tolerance)
                else:
                    self.assertLessEqual(objective, obj_misic + tolerance)
                    self.assertGreaterEqual(objective, obj_leaf - tolerance)
                if not ties_duplicated_threshold(case.tree_pairs, x_star, feas_tol):
                    self._assert_predict_reproduces(case, objective, x_star, feas_tol)

    def test_diabetes_agreement_ocean(self):
        self._check_between_misic_and_leaf("ocean")

    def test_diabetes_agreement_biggs_perakis(self):
        self._check_between_misic_and_leaf("biggs_perakis")


def _add_predictor_constr_silently(gpm, predictor, x, **kwargs):
    """``add_predictor_constr`` suppressing the global-epsilon warning.

    Used by tests that intentionally combine a positive epsilon with misic;
    the warning itself is asserted in
    ``test_positive_epsilon_warns_for_misic``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return add_predictor_constr(gpm, predictor, x, **kwargs)


def _two_split_chain(lower, upper):
    """Depth-2 tree ``root(lower) -> [0, node(upper) -> [0, 10]]``.

    Fitted for the shape, then the thresholds and leaf values are set
    directly: no fitted tree puts two thresholds closer together than the
    epsilon the tests use.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 5.0, 10.0])
    predictor = DecisionTreeRegressor(max_depth=2).fit(X, y)
    tree = predictor.tree_
    right = tree.children_right[0]
    tree.threshold[0] = lower
    tree.threshold[right] = upper
    tree.value[tree.children_left[0], 0, 0] = 0.0
    tree.value[tree.children_left[right], 0, 0] = 0.0
    tree.value[tree.children_right[right], 0, 0] = 10.0
    return predictor


class TestEpsilonAndFixedFeatures(unittest.TestCase):
    """Design decisions 2-3: epsilon semantics and fixed-feature handling."""

    def setUp(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        self.predictor = DecisionTreeRegressor(max_depth=1).fit(X, y)
        self.threshold = float(self.predictor.tree_.threshold[0])

    def test_epsilon_enforced_on_right_branch(self):
        self._check_epsilon_enforced_on_right_branch("leaf")

    def test_epsilon_enforced_on_right_branch_misic(self):
        self._check_epsilon_enforced_on_right_branch("misic")

    def test_epsilon_enforced_on_right_branch_ocean(self):
        self._check_epsilon_enforced_on_right_branch("ocean")

    def test_epsilon_enforced_on_right_branch_biggs_perakis(self):
        self._check_epsilon_enforced_on_right_branch("biggs_perakis")

    def test_lower_bound_in_band_skips_epsilon(self):
        self._check_epsilon_enforced_on_right_branch("leaf", lb_in_band=True)

    def test_lower_bound_in_band_skips_epsilon_misic(self):
        self._check_epsilon_enforced_on_right_branch("misic", lb_in_band=True)

    def test_lower_bound_in_band_skips_epsilon_ocean(self):
        self._check_epsilon_enforced_on_right_branch("ocean", lb_in_band=True)

    def test_lower_bound_in_band_skips_epsilon_biggs_perakis(self):
        self._check_epsilon_enforced_on_right_branch("biggs_perakis", lb_in_band=True)

    def _check_epsilon_enforced_on_right_branch(self, formulation, lb_in_band=False):
        """Selecting the right branch pushes x at least epsilon above the
        threshold when x can lie at or below it. When the lower bound of x
        is above the threshold, the bounds pick the branch and epsilon is
        skipped: x can go down to its lower bound."""
        epsilon = 1e-2
        lb = self.threshold + epsilon / 2.0 if lb_in_band else 0.0
        expected = lb if lb_in_band else self.threshold + epsilon
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, 1), lb=lb, ub=1.0)
            pred_constr = _add_predictor_constr_silently(
                gpm,
                self.predictor,
                x,
                epsilon=epsilon,
                formulation=formulation,
            )
            # Force the right leaf (value 1.0) and pull x down.
            gpm.addConstr(pred_constr.output.sum() >= 0.5)
            gpm.setObjective(x.sum(), GRB.MINIMIZE)
            gpm.optimize()
            self.assertEqual(gpm.Status, GRB.OPTIMAL)
            self.assertAlmostEqual(x.X[0, 0], expected, delta=gpm.Params.FeasibilityTol)

    def test_thresholds_closer_than_epsilon(self):
        self._check_thresholds_closer_than_epsilon("leaf")

    def test_thresholds_closer_than_epsilon_misic(self):
        self._check_thresholds_closer_than_epsilon("misic")

    def test_thresholds_closer_than_epsilon_ocean(self):
        self._check_thresholds_closer_than_epsilon("ocean")

    def test_thresholds_closer_than_epsilon_biggs_perakis(self):
        self._check_thresholds_closer_than_epsilon("biggs_perakis")

    def _check_thresholds_closer_than_epsilon(self, formulation):
        """Two thresholds closer together than epsilon must not forbid the
        branch above them: the right branch of the lower split stays
        reachable by clearing the upper threshold by epsilon."""
        epsilon = 1e-4
        lower, upper = 1.0 + 1e-6, 1.0 + 2e-6
        predictor = _two_split_chain(lower, upper)
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, 1), lb=1.0, ub=2.0)
            pred_constr = _add_predictor_constr_silently(
                gpm,
                predictor,
                x,
                epsilon=epsilon,
                formulation=formulation,
            )
            gpm.setObjective(pred_constr.output.sum(), GRB.MAXIMIZE)
            gpm.optimize()
            self.assertEqual(gpm.Status, GRB.OPTIMAL)
            self.assertAlmostEqual(pred_constr.output.X[0, 0], 10.0)
            self.assertGreaterEqual(
                x.X[0, 0], upper + epsilon - gpm.Params.FeasibilityTol
            )

    def test_fixed_feature_in_epsilon_band_is_feasible(self):
        self._check_fixed_feature_in_epsilon_band("leaf")

    def test_fixed_feature_in_epsilon_band_is_feasible_misic(self):
        self._check_fixed_feature_in_epsilon_band("misic")

    def test_fixed_feature_in_epsilon_band_is_feasible_ocean(self):
        self._check_fixed_feature_in_epsilon_band("ocean")

    def test_fixed_feature_in_epsilon_band_is_feasible_biggs_perakis(self):
        self._check_fixed_feature_in_epsilon_band("biggs_perakis")

    def _check_fixed_feature_in_epsilon_band(self, formulation):
        """A feature fixed to a constant inside ``(t, t + epsilon)`` must not
        make the model infeasible: the bounds pick the right branch, so
        epsilon is skipped."""
        epsilon = 1e-2
        value = self.threshold + epsilon / 2.0
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, 1), lb=value, ub=value)
            pred_constr = _add_predictor_constr_silently(
                gpm,
                self.predictor,
                x,
                epsilon=epsilon,
                formulation=formulation,
            )
            gpm.optimize()
            self.assertEqual(gpm.Status, GRB.OPTIMAL)
            self.assertAlmostEqual(pred_constr.output.X[0, 0], 1.0)

    def test_unfixed_box_inside_epsilon_band_reaches_right_leaf(self):
        self._check_unfixed_box_inside_epsilon_band("leaf")

    def test_unfixed_box_inside_epsilon_band_reaches_right_leaf_misic(self):
        self._check_unfixed_box_inside_epsilon_band("misic")

    def test_unfixed_box_inside_epsilon_band_reaches_right_leaf_ocean(self):
        self._check_unfixed_box_inside_epsilon_band("ocean")

    def test_unfixed_box_inside_epsilon_band_reaches_right_leaf_biggs_perakis(self):
        self._check_unfixed_box_inside_epsilon_band("biggs_perakis")

    def _check_unfixed_box_inside_epsilon_band(self, formulation):
        """A non-fixed input boxed strictly inside ``(t, t + epsilon)`` is
        above the threshold, so the bounds pick the right leaf and every
        value of the box stays feasible."""
        epsilon = 1e-2
        lb = self.threshold + epsilon / 4.0
        ub = self.threshold + epsilon / 2.0
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, 1), lb=lb, ub=ub)
            pred_constr = _add_predictor_constr_silently(
                gpm,
                self.predictor,
                x,
                epsilon=epsilon,
                formulation=formulation,
            )
            gpm.setObjective(x.sum(), GRB.MINIMIZE)
            gpm.optimize()
            self.assertEqual(gpm.Status, GRB.OPTIMAL)
            self.assertAlmostEqual(pred_constr.output.X[0, 0], 1.0)
            self.assertAlmostEqual(x.X[0, 0], lb, delta=gpm.Params.FeasibilityTol)

    def test_fixed_at_threshold_goes_left(self):
        self._check_fixed_at_threshold_goes_left("leaf")

    def test_fixed_at_threshold_goes_left_misic(self):
        self._check_fixed_at_threshold_goes_left("misic")

    def test_fixed_at_threshold_goes_left_ocean(self):
        self._check_fixed_at_threshold_goes_left("ocean")

    def test_fixed_at_threshold_goes_left_biggs_perakis(self):
        self._check_fixed_at_threshold_goes_left("biggs_perakis")

    def _check_fixed_at_threshold_goes_left(self, formulation):
        """An input fixed exactly at the threshold goes left, as in
        ``predict``, even when the objective prefers the right leaf."""
        epsilon = 1e-2
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, 1), lb=self.threshold, ub=self.threshold)
            pred_constr = _add_predictor_constr_silently(
                gpm,
                self.predictor,
                x,
                epsilon=epsilon,
                formulation=formulation,
            )
            gpm.setObjective(pred_constr.output.sum(), GRB.MAXIMIZE)
            gpm.optimize()
            self.assertEqual(gpm.Status, GRB.OPTIMAL)
            self.assertAlmostEqual(pred_constr.output.X[0, 0], 0.0)

    def test_epsilon_exemption_is_per_row_not_per_batch(self):
        """The "leaf" formulation used to exempt a feature from epsilon only
        when it was fixed (``lb == ub``) for every row of the batch at once;
        a single free row denied the exemption to every other row. It now
        decides the exemption per row, so a batch mixing a fixed row, a
        row decided only by one (possibly infinite) bound, and a genuinely
        free row must treat each of them independently."""
        epsilon = 1e-2
        fixed_in_band = self.threshold + epsilon / 2.0
        unbounded_decided_left = self.threshold - epsilon / 2.0
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar(
                (3, 1),
                lb=[[fixed_in_band], [-GRB.INFINITY], [0.0]],
                ub=[[fixed_in_band], [unbounded_decided_left], [1.0]],
            )
            pred_constr = _add_predictor_constr_silently(
                gpm,
                self.predictor,
                x,
                epsilon=epsilon,
                formulation="leaf",
            )
            # Force the free row (row 2) to the right leaf and pull it down;
            # it must still pay the full epsilon, regardless of the other
            # rows' exemption.
            gpm.addConstr(pred_constr.output[2, :].sum() >= 0.5)
            gpm.setObjective(x[2, :].sum(), GRB.MINIMIZE)
            gpm.optimize()
            self.assertEqual(gpm.Status, GRB.OPTIMAL)
            # Row 0: fixed inside the epsilon band, decided right without it.
            self.assertAlmostEqual(pred_constr.output.X[0, 0], 1.0)
            # Row 1: unbounded below, decided left by its upper bound alone.
            self.assertAlmostEqual(pred_constr.output.X[1, 0], 0.0)
            # Row 2: genuinely free, still pays the full epsilon.
            self.assertAlmostEqual(
                x.X[2, 0],
                self.threshold + epsilon,
                delta=gpm.Params.FeasibilityTol,
            )

    def test_positive_epsilon_warns_for_misic(self):
        """The shared binary split variables apply epsilon globally (every
        threshold of every tree, not only along the selected paths); passing
        a positive epsilon must warn. The other formulations stay silent."""
        params = {"OutputFlag": 0}
        for formulation in ("misic", "misic_lazy"):
            with (
                self.subTest(formulation=formulation),
                gp.Env(params=params) as env,
                gp.Model(env=env) as gpm,
            ):
                x = gpm.addMVar((1, 1), lb=0.0, ub=1.0)
                with self.assertWarnsRegex(UserWarning, "applies\\s+globally"):
                    add_predictor_constr(
                        gpm,
                        self.predictor,
                        x,
                        epsilon=1e-2,
                        formulation=formulation,
                    )
        for formulation, epsilon in (
            ("misic", 0.0),
            ("leaf", 1e-2),
            # biggs_perakis carries epsilon path-wise in its leaf boxes and
            # ocean in its flow linking, so the global-epsilon warning must
            # not fire for either.
            ("biggs_perakis", 1e-2),
            ("ocean", 1e-2),
        ):
            with (
                self.subTest(formulation=formulation, epsilon=epsilon),
                gp.Env(params=params) as env,
                gp.Model(env=env) as gpm,
            ):
                x = gpm.addMVar((1, 1), lb=0.0, ub=1.0)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    add_predictor_constr(
                        gpm,
                        self.predictor,
                        x,
                        epsilon=epsilon,
                        formulation=formulation,
                    )
                self.assertFalse([w for w in caught if "globally" in str(w.message)])


class TestLifecycle(unittest.TestCase):
    """``remove()`` restores the model; unknown formulations raise. The new
    formulations add variables through other paths (e.g. variables shared by
    the trees), so check that ``remove()`` still takes all of them."""

    def setUp(self):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        self.n_features = X.shape[1]
        self.predictors = [
            GradientBoostingRegressor(n_estimators=4, max_depth=2, random_state=0).fit(
                X, y
            ),
            RandomForestRegressor(n_estimators=4, max_depth=2, random_state=0).fit(
                X, y
            ),
            DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y),
        ]

    def test_add_remove_misic(self):
        self._add_remove("misic")

    def test_add_remove_ocean(self):
        self._add_remove("ocean")

    def test_add_remove_biggs_perakis(self):
        self._add_remove("biggs_perakis")

    def test_ocean_requires_finite_bounds(self):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((1, self.n_features), lb=-GRB.INFINITY)
            with self.assertRaisesRegex(ValueError, "finite bounds"):
                add_predictor_constr(gpm, self.predictors[0], x, formulation="ocean")

    def test_biggs_perakis_requires_finite_bounds(self):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((1, self.n_features), lb=-GRB.INFINITY)
            with self.assertRaisesRegex(ValueError, "finite bounds"):
                add_predictor_constr(
                    gpm, self.predictors[0], x, formulation="biggs_perakis"
                )

    def _add_remove(self, formulation):
        for predictor in self.predictors:
            with self.subTest(predictor=type(predictor).__name__):
                self._check_add_remove(predictor, formulation)

    def _check_add_remove(self, predictor, formulation):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((2, self.n_features), lb=-1000.0, ub=1000.0)
            y = gpm.addMVar((2, 1), lb=-GRB.INFINITY)
            gpm.update()
            numvars = gpm.NumVars

            pred_constr = add_predictor_constr(
                gpm, predictor, x, y, epsilon=EPSILON, formulation=formulation
            )
            self.assertEqual(gpm.NumVars, numvars + len(pred_constr.vars))
            self.assertEqual(gpm.NumConstrs, len(pred_constr.constrs))
            self.assertEqual(gpm.NumGenConstrs, len(pred_constr.genconstrs))

            pred_constr.remove()
            gpm.update()
            self.assertEqual(gpm.NumVars, numvars)
            self.assertEqual(gpm.NumConstrs, 0)
            self.assertEqual(gpm.NumGenConstrs, 0)

    def test_unknown_formulation(self):
        for predictor in self.predictors:
            with self.subTest(predictor=type(predictor).__name__), gp.Model() as gpm:
                gpm.Params.OutputFlag = 0
                x = gpm.addMVar((1, self.n_features), lb=-GRB.INFINITY)
                with self.assertRaisesRegex(ValueError, "Unknown formulation"):
                    add_predictor_constr(
                        gpm, predictor, x, formulation="no_such_formulation"
                    )


class TestModelSize(unittest.TestCase):
    """Criterion #4: with unbounded inputs (no pruning) the variable counts
    must match the size formulas of the formulation."""

    def test_misic_binaries_are_shared_split_variables(self):
        rng = np.random.RandomState(0)
        # Integer-valued features so split thresholds (midpoints) coincide
        # across trees and threshold sharing is actually exercised.
        X = rng.randint(0, 5, size=(200, 4)).astype(float)
        y = rng.uniform(size=200)
        nex = 2

        for n_estimators in (2, 6):
            predictor = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=0
            ).fit(X, y)

            trees = _sklearn_trees(predictor)
            thresholds = thresholds_by_feature(_tree_pairs(predictor))
            n_shared = sum(len(values) for values in thresholds.values())
            n_splits = sum(int((tree.children_left >= 0).sum()) for tree in trees)
            n_leaves = sum(int((tree.children_left < 0).sum()) for tree in trees)

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar(
                        (nex, X.shape[1]), lb=-GRB.INFINITY, ub=GRB.INFINITY
                    )
                    add_predictor_constr(gpm, predictor, x, formulation="misic")
                    gpm.update()

                    # One binary per example and distinct (feature, threshold)
                    # of the whole ensemble — not per tree.
                    self.assertEqual(gpm.NumBinVars, nex * n_shared)
                    self.assertLess(n_shared, n_splits)
                    # Total: input + output + split binaries + one continuous
                    # leaf variable per example and leaf.
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1] + nex + nex * n_shared + nex * n_leaves,
                    )
                    # Two indicator constraints link each unfixed binary.
                    self.assertEqual(gpm.NumGenConstrs, 2 * nex * n_shared)

    def test_ocean_binaries_are_only_branching_binaries(self):
        rng = np.random.RandomState(0)
        X = rng.randint(0, 5, size=(200, 4)).astype(float)
        y = rng.uniform(size=200)
        nex = 2

        for n_estimators in (2, 6):
            predictor = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=0
            ).fit(X, y)

            trees = _sklearn_trees(predictor)
            thresholds = thresholds_by_feature(_tree_pairs(predictor))
            n_intervals = sum(len(values) + 1 for values in thresholds.values())
            n_levels = sum(int(tree.max_depth) for tree in trees)
            n_nodes = sum(int(tree.node_count) for tree in trees)

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((nex, X.shape[1]), lb=-100.0, ub=100.0)
                    add_predictor_constr(gpm, predictor, x, formulation="ocean")
                    gpm.update()

                    # The paper's selling point: the only binaries are the
                    # branching variables — none per threshold — and there
                    # are no indicator constraints at all.
                    self.assertEqual(gpm.NumBinVars, nex * n_levels)
                    self.assertEqual(gpm.NumGenConstrs, 0)
                    # Total: input + output + branching binaries + one flow
                    # per example and node + one mu per example and interval.
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1]
                        + nex
                        + nex * n_levels
                        + nex * n_nodes
                        + nex * n_intervals,
                    )

    def test_biggs_perakis_binaries_are_leaf_variables(self):
        rng = np.random.RandomState(0)
        X = rng.randint(0, 5, size=(200, 4)).astype(float)
        y = rng.uniform(size=200)
        nex = 2

        for n_estimators in (2, 6):
            predictor = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=0
            ).fit(X, y)

            trees = _sklearn_trees(predictor)
            n_leaves = sum(int((tree.children_left < 0).sum()) for tree in trees)

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((nex, X.shape[1]), lb=-100.0, ub=100.0)
                    add_predictor_constr(gpm, predictor, x, formulation="biggs_perakis")
                    gpm.update()

                    # One binary leaf selector per example and leaf — like
                    # the naive leaf count — no indicator constraints and no
                    # shared variables of any kind. Each tree is formulated
                    # on its own, with its own output variable.
                    self.assertEqual(gpm.NumBinVars, nex * n_leaves)
                    self.assertEqual(gpm.NumGenConstrs, 0)
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1] + nex + nex * n_estimators + nex * n_leaves,
                    )


class TestPrintStats(unittest.TestCase):
    """The ensemble formulations have no per-tree sub-estimators: like a
    single decision tree, ``print_stats`` shows only the totals. The
    per-tree formulations ("leaf", "biggs_perakis") keep the per-estimator
    table."""

    def setUp(self):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        self.X = X
        self.predictor = GradientBoostingRegressor(
            n_estimators=3, max_depth=3, random_state=0
        ).fit(X, y)

    def _print_stats(self, formulation):
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar(
                (1, self.X.shape[1]),
                lb=self.X.min(axis=0),
                ub=self.X.max(axis=0),
            )
            pred_constr = add_predictor_constr(
                gpm, self.predictor, x, formulation=formulation
            )
            gpm.update()
            output = io.StringIO()
            pred_constr.print_stats(file=output)
            return output.getvalue(), gpm.NumBinVars

    def test_ensemble_formulations_print_no_estimator_table(self):
        for formulation in ("misic", "misic_lazy", "ocean"):
            with self.subTest(formulation=formulation):
                output, _ = self._print_stats(formulation)
                self.assertIn("Input has shape", output)
                # No dangling empty per-estimator table.
                self.assertNotIn("Estimator", output)

    def test_per_tree_formulations_keep_estimator_table(self):
        for formulation in ("leaf", "biggs_perakis"):
            with self.subTest(formulation=formulation):
                output, _ = self._print_stats(formulation)
                self.assertIn("Estimator", output)


class TestMisicLazy(unittest.TestCase):
    """``"misic_lazy"`` is ``"misic"`` with ``Lazy=3`` on the split-linking
    rows, and only on those."""

    def test_split_rows_are_lazy(self):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        predictors = [
            GradientBoostingRegressor(n_estimators=6, max_depth=3, random_state=0),
            RandomForestRegressor(n_estimators=4, max_depth=3, random_state=0),
            DecisionTreeRegressor(max_depth=3, random_state=0),
        ]
        nex = 2
        for predictor in predictors:
            predictor.fit(X, y)
            with self.subTest(predictor=type(predictor).__name__):
                n_splits = sum(
                    int((tree.children_left >= 0).sum())
                    for tree in _sklearn_trees(predictor)
                )
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    # Unbounded inputs: no pruning, every split links both
                    # of its subtrees.
                    x = gpm.addMVar((nex, X.shape[1]), lb=-GRB.INFINITY)
                    add_predictor_constr(gpm, predictor, x, formulation="misic_lazy")
                    gpm.update()
                    lazy = gpm.getAttr(GRB.Attr.Lazy, gpm.getConstrs())
                    self.assertEqual(lazy.count(3), nex * 2 * n_splits)
                    self.assertEqual(lazy.count(3) + lazy.count(0), len(lazy))


if __name__ == "__main__":
    unittest.main()
