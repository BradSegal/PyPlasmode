"""Integrated mechanism-specific recovery controls."""

import numpy as np
import pytest

import pyplasmode as ppm

pytestmark = pytest.mark.integration


def test_fractional_nomination_has_distinct_mechanism_estimands() -> None:
    """Exact, group, module, and interaction recovery retain separate meanings."""
    universe = ("a", "b", "c", "d")
    nomination = ppm.nomination_from_scores(universe, np.array([3.0, 2.0, 2.0, 0.0]), depth=2)
    base = dict(signal=np.arange(10, dtype=np.float64), feature_universe=universe)
    sparse = ppm.MaterializedTruth(
        kind="sparse",
        direct_features=("a", "b"),
        weights=(ppm.FeatureWeight("a", 1.0), ppm.FeatureWeight("b", 1.0)),
        **base,
    )
    group = ppm.MaterializedTruth(
        kind="correlated",
        admissible_groups=(ppm.RecoveryGroup("g", "b", ("b", "c")),),
        **base,
    )
    module = ppm.MaterializedTruth(
        kind="module",
        weights=(ppm.FeatureWeight("a", 1.0), ppm.FeatureWeight("d", 3.0)),
        **base,
    )
    interaction = ppm.MaterializedTruth(
        kind="interaction",
        direct_features=("b", "c"),
        **base,
    )

    assert ppm.evaluate_fractional_recovery(nomination, sparse, estimand="exact").value == 0.75
    assert ppm.evaluate_fractional_recovery(nomination, group, estimand="group").value == 1.0
    assert ppm.evaluate_fractional_recovery(nomination, module, estimand="module").value == 0.25
    assert (
        ppm.evaluate_fractional_recovery(nomination, interaction, estimand="interaction").value
        == 0.0
    )


def test_nonlinear_recovery_integrates_boundary_ties_without_replacement() -> None:
    """Group and interaction recovery use exact boundary inclusion probabilities."""
    universe = ("a", "b", "c", "d")
    nomination = ppm.nomination_from_scores(
        universe,
        np.array([3.0, 2.0, 2.0, 0.0]),
        depth=2,
    )
    base = dict(signal=np.arange(10, dtype=np.float64), feature_universe=universe)
    groups = ppm.MaterializedTruth(
        kind="correlated",
        admissible_groups=(
            ppm.RecoveryGroup("b", "b", ("b",)),
            ppm.RecoveryGroup("c", "c", ("c",)),
        ),
        **base,
    )
    interaction = ppm.MaterializedTruth(
        kind="interaction",
        direct_features=("b", "c"),
        **base,
    )

    assert ppm.evaluate_fractional_recovery(nomination, groups, estimand="group").value == 0.5
    assert (
        ppm.evaluate_fractional_recovery(
            nomination,
            interaction,
            estimand="interaction",
        ).value
        == 0.0
    )
