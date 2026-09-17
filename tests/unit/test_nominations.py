"""Component tests for support-aware biomarker nominations."""

import numpy as np
import pytest

import pyplasmode as ppm

pytestmark = pytest.mark.unit


def test_nomination_from_scores_conserves_boundary_tie_budget() -> None:
    """Fractional ties retain all positive boundary members without padding."""
    nomination = ppm.nomination_from_scores(
        ("a", "b", "c", "d"),
        np.array([3.0, 2.0, 2.0, 0.0]),
        depth=2,
    )

    assert nomination.membership.tolist() == [1.0, 0.5, 0.5, 0.0]
    assert nomination.positive_support == ("a", "b", "c")
    assert nomination.selected_features == ("a", "b", "c")
    assert nomination.achieved_size == 3
    assert nomination.membership_budget == 2.0


def test_nomination_does_not_pad_limited_positive_support() -> None:
    """Requested depth remains separate when fewer proteins have support."""
    nomination = ppm.nomination_from_scores(
        ("a", "b", "c"),
        np.array([2.0, 0.0, 0.0]),
        depth=3,
    )

    assert nomination.requested_depth == 3
    assert nomination.selected_features == ("a",)
    assert nomination.achieved_size == 1
    assert nomination.membership_budget == 1.0


def test_nomination_rejects_negative_or_nonfinite_scores() -> None:
    """Method-specific scores must already have the declared non-negative orientation."""
    for scores in (np.array([1.0, -1.0]), np.array([1.0, np.nan])):
        with pytest.raises(ValueError, match="scores"):
            ppm.nomination_from_scores(("a", "b"), scores, depth=1)


def test_nomination_rejects_non_tie_fractional_membership() -> None:
    """Public fractional membership must encode one uniform boundary tie."""
    with pytest.raises(ValueError, match="fractional nomination"):
        ppm.FractionalNomination(
            ("a", "b", "c"),
            np.array([1.0, 0.75, 0.25]),
            2,
            ("a", "b", "c"),
        )


def test_sampled_matched_nominations_preserve_support_ties_and_strata() -> None:
    """Truth-blind draws move observed membership geometry without changing margins."""
    observed = ppm.nomination_from_scores(
        ("a", "b", "c", "d", "e", "f"),
        np.array([4.0, 2.0, 2.0, 1.0, 0.0, 0.0]),
        depth=2,
    )
    strata = ("large", "small", "small", "large", "small", "small")

    draws = ppm.sample_matched_nominations(observed, strata=strata, draws=8, seed=17)

    assert len(draws) == 8
    assert all(item.achieved_size == observed.achieved_size for item in draws)
    assert any(item.positive_support != observed.positive_support for item in draws)
    assert all(
        np.array_equal(np.sort(item.membership), np.sort(observed.membership)) for item in draws
    )
    for item in draws:
        for stratum in set(strata):
            positions = np.asarray([value == stratum for value in strata])
            observed_support = sum(
                feature in observed.positive_support
                for feature, keep in zip(observed.feature_ids, positions, strict=True)
                if keep
            )
            sampled_support = sum(
                feature in item.positive_support
                for feature, keep in zip(item.feature_ids, positions, strict=True)
                if keep
            )
            assert sampled_support == observed_support
            assert np.array_equal(
                np.sort(item.membership[positions]),
                np.sort(observed.membership[positions]),
            )
