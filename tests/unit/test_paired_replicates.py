"""Adversarial identity, failure and denominator contracts for paired summaries."""

import numpy as np
import pytest

from pyplasmode import (
    FeatureGraph,
    compare_paired_replicates,
    evaluate_graph_region,
    summarize_replicates,
)

pytestmark = pytest.mark.unit


def test_permuted_pairs_preserve_differences_and_uncertainty() -> None:
    """Order cannot manufacture uncertainty for a constant world-wise effect."""
    result = compare_paired_replicates(
        ("a", "b", "c", "d"),
        [0.1, 0.3, 0.5, 0.7],
        ("d", "c", "b", "a"),
        [0.8, 0.6, 0.4, 0.2],
    )
    assert result.replicate_ids == ("a", "b", "c", "d")
    np.testing.assert_allclose(result.differences, [0.1] * 4)
    assert result.summary.mean == pytest.approx(0.1)
    assert result.summary.standard_error < 1e-15
    assert result.matched_pair_count == 4


def test_failed_and_unmatched_pairs_are_explicit_and_not_imputed() -> None:
    """Conditional complete-pair summaries retain all exclusion counts."""
    result = compare_paired_replicates(
        ("a", "b", "c", "d", "left"),
        [1, 2, None, 4, 10],
        ("right", "d", "c", "b", "a"),
        [10, None, 5, 4, 3],
        allow_unmatched=True,
    )
    assert result.replicate_ids == ("a", "b")
    assert result.summary.mean == 2
    assert result.summary.replicate_count == 2
    assert result.summary.failure_count == result.failed_pair_count == 2
    assert result.left_failure_count == result.right_failure_count == 1
    assert result.unmatched_left_count == result.unmatched_right_count == 1
    assert result.matched_pair_count == 4


@pytest.mark.parametrize(
    "ids, values",
    [
        (("a", "a"), [1, 2]),
        (("", "b"), [1, 2]),
        (("a", "b"), [1]),
        (("a", "b"), [1, np.nan]),
        (("a", "b"), [1, np.inf]),
    ],
)
def test_invalid_pairs_fail_before_inference(ids, values) -> None:
    with pytest.raises(ValueError):
        compare_paired_replicates(ids, values, ("a", "b"), [1, 2])


def test_mismatched_ids_require_explicit_intersection() -> None:
    with pytest.raises(ValueError, match="identities differ"):
        compare_paired_replicates(("a", "b"), [1, 2], ("a", "c"), [2, 3])


def test_one_successful_pair_cannot_claim_uncertainty() -> None:
    with pytest.raises(ValueError, match="two successful pairs"):
        compare_paired_replicates(("a", "b"), [1, None], ("b", "a"), [2, 3])


def test_edgeless_target_has_node_recovery_and_unavailable_edge_recovery() -> None:
    result = evaluate_graph_region(
        ("a", "b"),
        FeatureGraph(("a", "b"), ()),
        region=("a", "b"),
        depths=(1, 2),
    )
    assert result.at_depth[-1].node_coverage == 1
    assert all(row.edge_coverage is None and row.target_edge_count == 0 for row in result.at_depth)


def test_unsigned_decreases_preserve_effect_sign() -> None:
    result = compare_paired_replicates(
        ("a", "b"),
        np.array([2, 3], dtype=np.uint64),
        ("a", "b"),
        np.array([1, 2], dtype=np.uint64),
    )
    assert result.differences == (-1.0, -1.0)
    assert result.summary.mean == -1.0


@pytest.mark.parametrize("values", [[1 + 2j, 1 + 3j], [np.array([1]), 2]])
def test_complex_and_nonscalar_estimates_are_rejected(values) -> None:
    with pytest.raises(ValueError, match="real scalars"):
        compare_paired_replicates(("a", "b"), [0, 0], ("a", "b"), values)


def test_large_constant_estimates_have_finite_zero_uncertainty() -> None:
    result = summarize_replicates([1e308, 1e308])
    assert result.mean == result.confidence_interval_low == result.confidence_interval_high == 1e308
    assert result.standard_deviation == result.standard_error == 0


def test_unrepresentable_summary_is_rejected() -> None:
    with pytest.raises(ValueError, match="floating-point range"):
        summarize_replicates([-1e308, 1e308])


def test_replicate_summary_cannot_discard_imaginary_components() -> None:
    with pytest.raises(ValueError, match="real-valued"):
        summarize_replicates([1 + 2j, 1 + 3j])
