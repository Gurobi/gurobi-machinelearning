"""Shared helpers for the tree-ensemble formulation tests.

Framework-neutral machinery for the cross-formulation agreement tests:
probing the threshold cells a solution can represent, and detecting
knife-edge ties on thresholds duplicated across trees. Each framework's
test module provides its own extraction of the per-tree
``(feature, threshold)`` pairs — using the same threshold values (and
dtype) as the MIP formulation — and its own ``predict`` evaluation.
"""

import itertools

import numpy as np


def thresholds_by_feature(tree_pairs):
    """Collect each feature's split thresholds over all trees.

    Parameters
    ----------
    tree_pairs : list of set
        One set of ``(feature, threshold)`` pairs per tree.
    """
    thresholds = {}
    for pairs in tree_pairs:
        for feature, threshold in pairs:
            thresholds.setdefault(feature, set()).add(threshold)
    return thresholds


def representable_cells(x_star, thresholds, feas_tol):
    """Points of every threshold cell that x_star can represent.

    The solution's own cell, plus both adjacent cells for every coordinate
    that ties a split threshold. Probing both sides covers either branch the
    solver may have taken, without depending on how each framework breaks
    ties. The probes are one float32 ulp away from the threshold, since the
    frameworks compare in float32, where a float64 ulp would vanish.
    """
    candidates = []
    for feature, value in enumerate(x_star):
        coordinate_candidates = [value]
        for threshold in thresholds.get(feature, ()):
            tie_tol = max(feas_tol, float(np.spacing(np.float32(threshold))))
            if abs(value - threshold) <= tie_tol:
                below = float(np.nextafter(np.float32(threshold), np.float32(-np.inf)))
                above = float(np.nextafter(np.float32(threshold), np.float32(np.inf)))
                coordinate_candidates += [below, above]
        candidates.append(coordinate_candidates)
    return np.array(list(itertools.product(*candidates)))


def ties_duplicated_threshold(tree_pairs, x_star, feas_tol):
    """True if x_star ties a threshold that appears in more than one tree.

    Trees fitted on (resamples of) the same data reuse the exact same split
    thresholds. When ``x_star`` sits exactly on such a threshold, the leaf
    formulation — whose trees branch independently — may route different
    trees to different sides: a branch combination no real input reproduces.
    Optima equality across formulations is only asserted without such a tie.
    """
    seen, duplicated = set(), set()
    for pairs in tree_pairs:
        duplicated |= pairs & seen
        seen |= pairs
    for feature, threshold in duplicated:
        tie_tol = max(feas_tol, float(np.spacing(np.float32(threshold))))
        if abs(x_star[feature] - threshold) <= tie_tol:
            return True
    return False


class ProjectInputMixin:
    """Tests of ``project_input`` shared by the frameworks.

    The test class provides ``X``, ``predictor`` (a fitted regressor whose
    ``predict`` takes a numpy array) and derives from ``unittest.TestCase``.
    """

    FORMULATIONS = ("leaf", "misic", "misic_lazy", "ocean", "biggs_perakis")
    #: Large enough for Gurobi's tolerances and the float32 rounding of the
    #: features of the diabetes data.
    EPSILON = 1e-4

    def solve(self, formulation, epsilon):
        """Maximize the prediction of two inputs; return the value of the
        output variables and the projected inputs."""
        import warnings

        import gurobipy as gp

        from gurobi_ml import add_predictor_constr

        with gp.Model() as model:
            model.Params.OutputFlag = 0
            x = model.addMVar(
                (2, self.X.shape[1]), lb=self.X.min(axis=0), ub=self.X.max(axis=0)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pred_constr = add_predictor_constr(
                    model, self.predictor, x, formulation=formulation, epsilon=epsilon
                )
            model.setObjective(pred_constr.output.sum(), gp.GRB.MAXIMIZE)
            model.optimize()
            self.assertEqual(model.Status, gp.GRB.OPTIMAL)
            return pred_constr.output.X.ravel(), pred_constr.project_input()

    def test_projection_attains_output_with_epsilon(self):
        # With a sufficient epsilon, the selected leaves of every
        # formulation contain an input, and the regression evaluated at the
        # projection gives the values of the output variables.
        for formulation in self.FORMULATIONS:
            with self.subTest(formulation=formulation):
                output, projected = self.solve(formulation, self.EPSILON)
                self.assertFalse(np.isnan(projected).any())
                np.testing.assert_allclose(
                    self.predictor.predict(projected).ravel(), output, rtol=1e-6
                )

    def test_projection_attains_output_misic_without_epsilon(self):
        # The shared split variables keep the leaves compatible at epsilon=0.
        for formulation in ("misic", "misic_lazy"):
            with self.subTest(formulation=formulation):
                output, projected = self.solve(formulation, 0.0)
                self.assertFalse(np.isnan(projected).any())
                np.testing.assert_allclose(
                    self.predictor.predict(projected).ravel(), output, rtol=1e-6
                )

    def test_projection_without_epsilon_attains_output_or_is_nan(self):
        # At epsilon=0 the other formulations may select leaves that no
        # input reaches: such rows are NaN, the others attain the output.
        for formulation in ("leaf", "ocean", "biggs_perakis"):
            with self.subTest(formulation=formulation):
                output, projected = self.solve(formulation, 0.0)
                reached = ~np.isnan(projected).any(axis=1)
                if reached.any():
                    np.testing.assert_allclose(
                        self.predictor.predict(projected[reached]).ravel(),
                        output[reached],
                        rtol=1e-6,
                    )

    def test_no_solution(self):
        import gurobipy as gp

        from gurobi_ml import add_predictor_constr
        from gurobi_ml.exceptions import NoSolutionError

        with gp.Model() as model:
            x = model.addMVar((1, self.X.shape[1]), lb=-1, ub=1)
            pred_constr = add_predictor_constr(model, self.predictor, x)
            with self.assertRaises(NoSolutionError):
                pred_constr.project_input()
