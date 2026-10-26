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
