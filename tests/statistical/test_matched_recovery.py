"""Statistical reference tests for analytic matched biomarker recovery."""

from math import comb

import numpy as np
import pytest
from scipy.stats import hypergeom

import pyplasmode as ppm

pytestmark = pytest.mark.statistical


def _truth(kind: str = "sparse") -> ppm.MaterializedTruth:
    features = tuple(f"p{index}" for index in range(10))
    return ppm.MaterializedTruth(
        kind=kind,  # type: ignore[arg-type]
        signal=np.arange(20, dtype=np.float64),
        feature_universe=features,
        direct_features=("p0", "p1"),
        weights=(ppm.FeatureWeight("p0", 1.0), ppm.FeatureWeight("p1", 1.0)),
    )


def _sampled_expectation(
    nomination: ppm.FractionalNomination,
    truth: ppm.MaterializedTruth,
    *,
    estimand: str,
    strata: tuple[str, ...],
    draws: int = 40_000,
) -> float:
    references = ppm.sample_matched_nominations(
        nomination,
        strata=strata,
        draws=draws,
        seed=20260903,
    )
    return float(
        np.mean(
            [
                ppm.evaluate_fractional_recovery(
                    reference,
                    truth,
                    estimand=estimand,  # type: ignore[arg-type]
                ).value
                for reference in references
            ]
        )
    )


def test_sparse_expectation_agrees_with_scipy_hypergeometric() -> None:
    """Analytic exact recovery matches an independent maintained distribution."""
    truth = _truth()
    nomination = ppm.nomination_from_scores(
        truth.feature_universe,
        np.array([10.0, 0.0, 9.0, 8.0, 0.0, 7.0, 6.0, 5.0, 4.0, 3.0]),
        depth=2,
    )
    result = ppm.evaluate_matched_recovery(
        nomination,
        truth,
        estimand="exact",
        strata=("all",) * 10,
    )

    expected_true_count = float(hypergeom.mean(10, 2, 2))
    expected_recovery = expected_true_count / 2.0
    assert result.method == "analytic"
    assert result.admissible_panel_count == 1_260
    assert result.expected_recovery == pytest.approx(expected_recovery)
    assert result.observed_recovery == 0.5
    assert result.chance_adjusted_recovery == pytest.approx(0.5 - expected_recovery)


def test_single_matched_panel_retains_defined_zero_contrast() -> None:
    """A fixed matched reference remains descriptive rather than becoming missing."""
    truth = _truth()
    nomination = ppm.nomination_from_scores(
        truth.feature_universe,
        np.array([2.0, 1.0] + [0.0] * 8),
        depth=2,
    )
    result = ppm.evaluate_matched_recovery(
        nomination,
        truth,
        estimand="exact",
        strata=("selected_0", "selected_1", *(f"other_{index}" for index in range(8))),
    )

    assert result.admissible_panel_count == 1
    assert result.expected_recovery == result.observed_recovery == 1.0
    assert result.chance_adjusted_recovery == 0.0


def test_stratified_exact_expectation_matches_sampled_reference() -> None:
    """Analytic expectation respects different support margins across strata."""
    truth = _truth()
    nomination = ppm.nomination_from_scores(
        truth.feature_universe,
        np.asarray([8.0, 0.0, 7.0, 0.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        depth=3,
    )
    strata = ("small",) * 4 + ("large",) * 6
    result = ppm.evaluate_matched_recovery(
        nomination,
        truth,
        estimand="exact",
        strata=strata,
    )

    sampled = _sampled_expectation(
        nomination,
        truth,
        estimand="exact",
        strata=strata,
    )
    assert result.expected_recovery == pytest.approx(sampled, abs=0.01)


def test_group_tie_expectation_agrees_with_hypergeometric_reference() -> None:
    """Nonlinear group recovery integrates the random tied boundary exactly."""
    universe = tuple(f"p{index}" for index in range(6))
    nomination = ppm.nomination_from_scores(
        universe,
        np.array([4.0, 3.0, 3.0, 0.0, 0.0, 0.0]),
        depth=2,
    )
    truth = ppm.MaterializedTruth(
        kind="correlated",
        signal=np.arange(20, dtype=np.float64),
        feature_universe=universe,
        admissible_groups=(
            ppm.RecoveryGroup("p0", "p0", ("p0",)),
            ppm.RecoveryGroup("p1", "p1", ("p1",)),
        ),
    )

    result = ppm.evaluate_matched_recovery(
        nomination,
        truth,
        estimand="group",
        strata=("all",) * len(universe),
    )

    expected = float(hypergeom.mean(len(universe), 2, 2) / 2.0)
    assert result.observed_recovery == 0.75
    assert result.expected_recovery == pytest.approx(expected)


def test_stratified_group_expectation_matches_sampled_reference() -> None:
    """Group localisation integrates stratum assignment and a global boundary tie."""
    universe = tuple(f"p{index}" for index in range(10))
    nomination = ppm.nomination_from_scores(
        universe,
        np.asarray([9.0, 8.0, 7.0, 7.0, 0.0, 6.0, 5.0, 4.0, 4.0, 0.0]),
        depth=5,
    )
    truth = ppm.MaterializedTruth(
        kind="correlated",
        signal=np.arange(20, dtype=np.float64),
        feature_universe=universe,
        admissible_groups=(
            ppm.RecoveryGroup("p0", "p0", ("p0", "p2", "p6")),
            ppm.RecoveryGroup("p5", "p5", ("p5", "p8")),
        ),
    )
    strata = ("a",) * 5 + ("b",) * 5

    result = ppm.evaluate_matched_recovery(
        nomination,
        truth,
        estimand="group",
        strata=strata,
    )
    sampled = _sampled_expectation(
        nomination,
        truth,
        estimand="group",
        strata=strata,
    )

    assert result.expected_recovery == pytest.approx(sampled, abs=0.01)


def test_interaction_tie_expectation_agrees_with_combinatorial_reference() -> None:
    """Interaction recovery uses joint inclusion rather than marginal products."""
    universe = tuple(f"p{index}" for index in range(6))
    nomination = ppm.nomination_from_scores(
        universe,
        np.array([4.0, 3.0, 3.0, 0.0, 0.0, 0.0]),
        depth=2,
    )
    truth = ppm.MaterializedTruth(
        kind="interaction",
        signal=np.arange(20, dtype=np.float64),
        feature_universe=universe,
        direct_features=("p1", "p2"),
    )

    result = ppm.evaluate_matched_recovery(
        nomination,
        truth,
        estimand="interaction",
        strata=("all",) * len(universe),
    )

    assert result.observed_recovery == 0.0
    assert result.expected_recovery == pytest.approx(1.0 / comb(len(universe), 2))


def test_weighted_module_expectation_uses_marginal_selection_probability() -> None:
    """Weighted module recovery preserves the matched selection budget."""
    universe = tuple(f"p{index}" for index in range(8))
    nomination = ppm.nomination_from_scores(
        universe,
        np.arange(8.0, 0.0, -1.0),
        depth=3,
    )
    truth = ppm.MaterializedTruth(
        kind="module",
        signal=np.arange(20, dtype=np.float64),
        feature_universe=universe,
        weights=(
            ppm.FeatureWeight("p0", 1.0),
            ppm.FeatureWeight("p1", 2.0),
            ppm.FeatureWeight("p2", 3.0),
        ),
    )

    result = ppm.evaluate_matched_recovery(
        nomination,
        truth,
        estimand="module",
        strata=("all",) * len(universe),
    )

    assert result.expected_recovery == pytest.approx(3.0 / len(universe))


def test_outcome_specificity_compares_own_with_disjoint_foreign_truths() -> None:
    """Specificity is the paired own-minus-foreign adjusted recovery contrast."""
    universe = tuple(f"p{index}" for index in range(10))

    def sparse(left: str, right: str) -> ppm.MaterializedTruth:
        return ppm.MaterializedTruth(
            kind="sparse",
            signal=np.arange(20, dtype=np.float64),
            feature_universe=universe,
            direct_features=(left, right),
            weights=(ppm.FeatureWeight(left, 1.0), ppm.FeatureWeight(right, 1.0)),
        )

    nomination = ppm.nomination_from_scores(
        universe,
        np.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        depth=2,
    )
    result = ppm.evaluate_outcome_specificity(
        nomination,
        sparse("p0", "p1"),
        (sparse("p2", "p3"), sparse("p4", "p5")),
        estimand="exact",
        strata=("all",) * len(universe),
    )

    assert result.own_adjusted_recovery == pytest.approx(0.8)
    assert result.mean_foreign_adjusted_recovery == pytest.approx(-0.2)
    assert result.specificity == pytest.approx(1.0)
    assert result.foreign_truth_count == 2


def test_outcome_specificity_rejects_overlapping_truth_targets() -> None:
    """A shared credited feature invalidates the own-versus-foreign contrast."""
    truth = _truth()
    foreign = ppm.MaterializedTruth(
        kind="sparse",
        signal=np.arange(20, dtype=np.float64),
        feature_universe=truth.feature_universe,
        direct_features=("p1", "p2"),
    )
    nomination = ppm.nomination_from_scores(
        truth.feature_universe,
        np.arange(10.0, 0.0, -1.0),
        depth=2,
    )

    with pytest.raises(ValueError, match="not disjoint"):
        ppm.evaluate_outcome_specificity(
            nomination,
            truth,
            (foreign,),
            estimand="exact",
            strata=("all",) * 10,
        )
