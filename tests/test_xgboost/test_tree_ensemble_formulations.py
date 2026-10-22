"""Tests for the tree ensemble formulations with XGBoost.

Mirrors the cross-formulation agreement, lifecycle and model-size tests of
``tests/test_sklearn/test_tree_ensemble_formulations.py`` (where the checks
are documented) for the XGBoost wiring.

XGBoost branches left iff ``x < t`` (strict) and its thresholds sit at
(float32) data values; the formulations move the left side by epsilon
instead of the right one. At the default ``epsilon = 0`` the comparisons
below share the closure semantics of the other frameworks.
"""

import io
import json
import unittest
import warnings

import gurobipy as gp
import numpy as np
import xgboost as xgb
from gurobipy import GRB, GurobiError
from sklearn import datasets

from gurobi_ml import add_predictor_constr

from ..tree_ensemble import (
    representable_cells,
    thresholds_by_feature,
    ties_duplicated_threshold,
)

EPSILON = 0.0


def _tree_pairs(regressor):
    """The (feature, threshold) pairs of each tree of the booster.

    Thresholds are read exactly as the extraction does (float32) so that
    cell probing and tie detection see the MIP's values.
    """
    raw = json.loads(regressor.get_booster().save_raw(raw_format="json"))
    pairs = []
    for tree in raw["learner"]["gradient_booster"]["model"]["trees"]:
        split_nodes = np.array(tree["left_children"]) >= 0
        features = np.array(tree["split_indices"])[split_nodes]
        thresholds = np.array(tree["split_conditions"], dtype=np.float32)[split_nodes]
        pairs.append(set(zip(features.tolist(), thresholds.tolist())))
    return pairs


class TestCrossFormulationAgreement(unittest.TestCase):
    """Cross-formulation agreement for XGBoost regressors (reg:squarederror),
    see the sklearn test module. ``setUpClass`` solves every formulation once
    per sense; the tests only compare the solutions."""

    FORMULATIONS = ("leaf", "misic", "ocean", "biggs_perakis")

    @classmethod
    def setUpClass(cls):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        cls.predictor = xgb.XGBRegressor(
            n_estimators=5, max_depth=3, random_state=0
        ).fit(X, y)
        # Per tree: for the duplicated-threshold ties. Per feature: for the
        # cells a solution can represent.
        cls.tree_pairs = _tree_pairs(cls.predictor)
        cls.thresholds = thresholds_by_feature(cls.tree_pairs)
        cls.solutions = {
            sense: {
                formulation: cls._optimize(cls.predictor, X, formulation, sense)
                for formulation in cls.FORMULATIONS
            }
            for sense in (GRB.MAXIMIZE, GRB.MINIMIZE)
        }

    @staticmethod
    def _optimize(predictor, X, formulation, sense):
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

    def _solution(self, sense, formulation):
        status, max_vio, feas_tol, objective, x_star = self.solutions[sense][
            formulation
        ]
        self.assertEqual(status, GRB.OPTIMAL, msg=formulation)
        self.assertLessEqual(max_vio, feas_tol, msg=formulation)
        return objective, x_star, feas_tol

    def _assert_predict_reproduces(self, objective, x_star, feas_tol):
        cells = representable_cells(x_star, self.thresholds, feas_tol)
        predictions = self.predictor.predict(cells)
        tolerance = 1e-4 * (1.0 + abs(objective))
        self.assertLessEqual(
            np.min(np.abs(predictions - objective)),
            tolerance,
            msg="MIP optimum not reproduced by predict on any representable cell",
        )

    def test_diabetes_agreement_misic(self):
        """The misic optimum is reproduced by predict and bounded by the leaf
        optimum; without a duplicated-threshold tie the two agree."""
        for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
            with self.subTest(sense=sense):
                obj_leaf, x_leaf, feas_tol = self._solution(sense, "leaf")
                objective, x_star, _ = self._solution(sense, "misic")
                # The shared z couples the trees, so a misic optimum
                # always corresponds to a real input.
                self._assert_predict_reproduces(objective, x_star, feas_tol)

                # Each solve is proven optimal within the default
                # relative MIPGap of 1e-4.
                tolerance = 3e-4 * max(1.0, abs(objective))

                # Every misic-feasible point is leaf-feasible, so the
                # leaf optimum dominates.
                if sense == GRB.MAXIMIZE:
                    self.assertGreaterEqual(obj_leaf, objective - tolerance)
                else:
                    self.assertLessEqual(obj_leaf, objective + tolerance)

                # Without a duplicated-threshold tie the leaf formulation
                # cannot mix branches across trees.
                if not ties_duplicated_threshold(self.tree_pairs, x_leaf, feas_tol):
                    self.assertLessEqual(abs(obj_leaf - objective), tolerance)
                    self._assert_predict_reproduces(obj_leaf, x_leaf, feas_tol)

    def _check_between_misic_and_leaf(self, formulation):
        """Sandwich misic <= formulation <= leaf (maximize) — see the
        sklearn test module."""
        for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
            with self.subTest(sense=sense):
                obj_leaf, _, feas_tol = self._solution(sense, "leaf")
                obj_misic, _, _ = self._solution(sense, "misic")
                objective, x_star, _ = self._solution(sense, formulation)
                tolerance = 3e-4 * max(1.0, abs(objective))
                if sense == GRB.MAXIMIZE:
                    self.assertGreaterEqual(objective, obj_misic - tolerance)
                    self.assertLessEqual(objective, obj_leaf + tolerance)
                else:
                    self.assertLessEqual(objective, obj_misic + tolerance)
                    self.assertGreaterEqual(objective, obj_leaf - tolerance)
                if not ties_duplicated_threshold(self.tree_pairs, x_star, feas_tol):
                    self._assert_predict_reproduces(objective, x_star, feas_tol)

    def test_diabetes_agreement_ocean(self):
        self._check_between_misic_and_leaf("ocean")

    def test_diabetes_agreement_biggs_perakis(self):
        self._check_between_misic_and_leaf("biggs_perakis")


class TestLifecycle(unittest.TestCase):
    """``remove()`` restores the model; unknown formulations raise; a
    positive epsilon warns for misic. The new formulations add variables
    through other paths (e.g. variables shared by the trees), so check that
    ``remove()`` still takes all of them."""

    def setUp(self):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        self.n_features = X.shape[1]
        self.predictor = xgb.XGBRegressor(
            n_estimators=4, max_depth=2, random_state=0
        ).fit(X, y)

    def test_add_remove_misic(self):
        self._check_add_remove("misic")

    def test_add_remove_ocean(self):
        self._check_add_remove("ocean")

    def test_add_remove_biggs_perakis(self):
        self._check_add_remove("biggs_perakis")

    def _check_add_remove(self, formulation):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((2, self.n_features), lb=-1000.0, ub=1000.0)
            y = gpm.addMVar((2, 1), lb=-GRB.INFINITY)
            gpm.update()
            numvars = gpm.NumVars

            pred_constr = add_predictor_constr(
                gpm, self.predictor, x, y, formulation=formulation
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
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((1, self.n_features), lb=-GRB.INFINITY)
            with self.assertRaisesRegex(ValueError, "Unknown formulation"):
                add_predictor_constr(
                    gpm, self.predictor, x, formulation="no_such_formulation"
                )

    def test_positive_epsilon_warns_for_misic(self):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((1, self.n_features), lb=-GRB.INFINITY)
            with self.assertWarnsRegex(UserWarning, "applies globally"):
                add_predictor_constr(
                    gpm, self.predictor, x, epsilon=1e-5, formulation="misic"
                )


class TestStrictLeftEpsilon(unittest.TestCase):
    """XGBoost sends ``x < t`` left, so epsilon moves the left side of a
    split: going left needs ``x <= t - epsilon``, unless the input bounds
    already pick the branch."""

    FORMULATIONS = ("leaf", "misic", "ocean", "biggs_perakis")
    EPSILON = 1e-2

    def setUp(self):
        X = np.array([[0.0], [1.0]] * 5)
        y = np.array([0.0, 1.0] * 5)
        self.predictor = xgb.XGBRegressor(
            n_estimators=1,
            max_depth=1,
            learning_rate=1.0,
            min_child_weight=0,
            reg_lambda=0,
            base_score=0.5,
        ).fit(X, y)
        raw = json.loads(self.predictor.get_booster().save_raw(raw_format="json"))
        tree = raw["learner"]["gradient_booster"]["model"]["trees"][0]
        self.threshold = float(tree["split_conditions"][0])
        self.left_value, self.right_value = self.predictor.predict(
            np.array([[self.threshold - 1.0], [self.threshold]], dtype=np.float32)
        )

    def _solve(self, formulation, lb, ub, objective, sense):
        """Build the stump on x in [lb, ub], optimize, return (x, output)."""
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, 1), lb=lb, ub=ub)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pred_constr = add_predictor_constr(
                    gpm,
                    self.predictor,
                    x,
                    epsilon=self.EPSILON,
                    formulation=formulation,
                )
            gpm.setObjective(objective(x, pred_constr.output), sense)
            gpm.optimize()
            self.assertEqual(gpm.Status, GRB.OPTIMAL)
            return x.X[0, 0], pred_constr.output.X[0, 0]

    def test_fixed_in_band_goes_left(self):
        """A value fixed in ``(t - epsilon, t)`` goes left, as in predict."""
        value = self.threshold - self.EPSILON / 2.0
        for formulation in self.FORMULATIONS:
            with self.subTest(formulation=formulation):
                _, output = self._solve(
                    formulation, value, value, lambda x, y: y.sum(), GRB.MAXIMIZE
                )
                self.assertAlmostEqual(output, self.left_value, places=5)

    def test_box_inside_band_reaches_left_leaf(self):
        """A box inside ``(t - epsilon, t)`` is below the threshold: the bounds
        pick the left leaf and every value of the box stays feasible."""
        lb = self.threshold - self.EPSILON / 2.0
        ub = self.threshold - self.EPSILON / 4.0
        for formulation in self.FORMULATIONS:
            with self.subTest(formulation=formulation):
                x, output = self._solve(
                    formulation, lb, ub, lambda x, y: x.sum(), GRB.MAXIMIZE
                )
                self.assertAlmostEqual(output, self.left_value, places=5)
                self.assertAlmostEqual(x, ub, delta=1e-6)

    def test_epsilon_enforced_on_left_branch(self):
        """When x can lie on both sides, going left needs x <= t - epsilon."""
        for formulation in self.FORMULATIONS:
            with self.subTest(formulation=formulation):
                x, _ = self._solve(
                    formulation,
                    0.0,
                    2.0,
                    lambda x, y: x.sum() - 10.0 * y.sum(),
                    GRB.MAXIMIZE,
                )
                self.assertLessEqual(x, self.threshold - self.EPSILON + 1e-6)


class TestModelSize(unittest.TestCase):
    """With unbounded inputs the binary count equals the number of distinct
    (feature, threshold) pairs of the ensemble — not one copy per tree."""

    def test_misic_binaries_are_shared_split_variables(self):
        rng = np.random.RandomState(0)
        # Integer-valued features so split thresholds coincide across trees
        # and threshold sharing is actually exercised.
        X = rng.randint(0, 5, size=(200, 4)).astype(float)
        y = rng.uniform(size=200)
        nex = 2

        for n_estimators in (2, 6):
            predictor = xgb.XGBRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=0
            ).fit(X, y)

            tree_pairs = _tree_pairs(predictor)
            n_shared = sum(
                len(values) for values in thresholds_by_feature(tree_pairs).values()
            )
            n_splits = sum(len(pairs) for pairs in tree_pairs)
            raw = json.loads(predictor.get_booster().save_raw(raw_format="json"))
            n_leaves = sum(
                int((np.array(tree["left_children"]) < 0).sum())
                for tree in raw["learner"]["gradient_booster"]["model"]["trees"]
            )

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar(
                        (nex, X.shape[1]), lb=-GRB.INFINITY, ub=GRB.INFINITY
                    )
                    add_predictor_constr(gpm, predictor, x, formulation="misic")
                    gpm.update()

                    self.assertEqual(gpm.NumBinVars, nex * n_shared)
                    self.assertLess(n_shared, n_splits)
                    # Total: input + output + split binaries + one
                    # continuous leaf variable per example and leaf. The
                    # ensemble's sum of trees is an expression, not a
                    # variable.
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1] + nex + nex * n_shared + nex * n_leaves,
                    )
                    self.assertEqual(gpm.NumGenConstrs, 2 * nex * n_shared)

    def test_ocean_binaries_are_only_branching_binaries(self):
        rng = np.random.RandomState(0)
        X = rng.randint(0, 5, size=(200, 4)).astype(float)
        y = rng.uniform(size=200)
        nex = 2

        for n_estimators in (2, 6):
            predictor = xgb.XGBRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=0
            ).fit(X, y)

            thresholds = thresholds_by_feature(_tree_pairs(predictor))
            n_intervals = sum(len(values) + 1 for values in thresholds.values())
            raw = json.loads(predictor.get_booster().save_raw(raw_format="json"))
            n_nodes = 0
            n_levels = 0
            for tree in raw["learner"]["gradient_booster"]["model"]["trees"]:
                left = np.array(tree["left_children"])
                n_nodes += left.size
                depth = np.zeros(left.size, dtype=int)
                right = np.array(tree["right_children"])
                for node in range(left.size):
                    if left[node] >= 0:
                        depth[left[node]] = depth[right[node]] = depth[node] + 1
                n_levels += int(depth[left >= 0].max()) + 1

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((nex, X.shape[1]), lb=-100.0, ub=100.0)
                    add_predictor_constr(gpm, predictor, x, formulation="ocean")
                    gpm.update()

                    self.assertEqual(gpm.NumBinVars, nex * n_levels)
                    self.assertEqual(gpm.NumGenConstrs, 0)
                    # Total: input + output + branching binaries + flows +
                    # mu intervals. The ensemble's sum of trees is an
                    # expression, not a variable.
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
            predictor = xgb.XGBRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=0
            ).fit(X, y)

            raw = json.loads(predictor.get_booster().save_raw(raw_format="json"))
            n_leaves = sum(
                int((np.array(tree["left_children"]) < 0).sum())
                for tree in raw["learner"]["gradient_booster"]["model"]["trees"]
            )

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((nex, X.shape[1]), lb=-100.0, ub=100.0)
                    add_predictor_constr(gpm, predictor, x, formulation="biggs_perakis")
                    gpm.update()

                    self.assertEqual(gpm.NumBinVars, nex * n_leaves)
                    self.assertEqual(gpm.NumGenConstrs, 0)
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1] + nex + nex * n_estimators + nex * n_leaves,
                    )


class TestPrintStats(unittest.TestCase):
    """``print_stats`` shows no per-estimator table for an ensemble
    formulation (mirrors the sklearn test, where the checks are
    documented)."""

    def test_misic_prints_no_estimator_table(self):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        predictor = xgb.XGBRegressor(n_estimators=3, max_depth=3, random_state=0).fit(
            X, y
        )
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, X.shape[1]), lb=X.min(axis=0), ub=X.max(axis=0))
            pred_constr = add_predictor_constr(gpm, predictor, x, formulation="misic")
            output = io.StringIO()
            pred_constr.print_stats(file=output)
        self.assertIn("Input has shape", output.getvalue())
        self.assertNotIn("Estimator", output.getvalue())


class TestMisicLazy(unittest.TestCase):
    """``"misic_lazy"`` marks only split-linking rows lazy (mirrors the
    sklearn test, where the checks are documented)."""

    def test_split_rows_are_lazy(self):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        predictor = xgb.XGBRegressor(n_estimators=3, max_depth=3, random_state=0).fit(
            X, y
        )
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, X.shape[1]), lb=X.min(axis=0), ub=X.max(axis=0))
            add_predictor_constr(gpm, predictor, x, formulation="misic_lazy")
            gpm.update()
            lazy = gpm.getAttr(GRB.Attr.Lazy, gpm.getConstrs())
            self.assertGreater(lazy.count(3), 0)
            self.assertEqual(lazy.count(3) + lazy.count(0), len(lazy))


if __name__ == "__main__":
    unittest.main()
