"""Hand calculations and public boundary checks for the metrics module."""

from itertools import permutations, product

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pyplasmode import nomination_from_scores, rank_stability
from pyplasmode.metrics import (
    consensus_ranking,
    cumulative_dynamic_auc,
    learnability_gap,
    paired_bootstrap_difference,
    rank_biased_overlap,
    selection_stability,
)

pytestmark = pytest.mark.unit


def test_nogueira_hand_calculation_and_degenerate_selections():
    names = tuple("abcd")
    rows = [[1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0]]
    nominations = [nomination_from_scores(names, row, depth=2) for row in rows]
    matrix = np.asarray(rows)
    fraction = matrix.mean()
    expected = 1 - matrix.var(axis=0, ddof=1).mean() / (fraction * (1 - fraction))
    result = selection_stability(nominations)
    assert result.value == pytest.approx(expected)
    assert result.membership_budgets == (2, 1, 2)
    for row in ([0] * 4, [1] * 4):
        item = nomination_from_scores(names, row, depth=4)
        assert selection_stability([item, item]).value is None
    fixed = nomination_from_scores(names, rows[0], depth=2)
    assert selection_stability([fixed, fixed]).value == 1


def test_tie_stability_matches_exhaustive_independent_resolution():
    names = tuple("abcd")
    tied = nomination_from_scores(names, [1, 1, 1, 0], depth=1)
    resolved = [nomination_from_scores(names, row, depth=1) for row in np.eye(4)[:3]]
    expected = np.mean([selection_stability(pair).value for pair in product(resolved, repeat=2)])
    assert selection_stability([tied, tied]).value == pytest.approx(expected)
    uninformative = nomination_from_scores(names, [1] * 4, depth=2)
    assert selection_stability([uninformative] * 20).value == pytest.approx(0)
    reordered = nomination_from_scores(tuple("dcba"), [0, 1, 1, 1], depth=1)
    assert selection_stability([tied, reordered]).value == pytest.approx(expected)
    with pytest.raises(ValueError, match="universe"):
        selection_stability([tied, nomination_from_scores(tuple("abce"), [1] * 4, depth=1)])
    with pytest.raises(ValueError, match="two"):
        selection_stability([tied])


def test_rbo_matches_all_tie_permutations_and_existing_untied_metric():
    names = tuple("abcd")
    left = [1, 1, 3, 4]
    right = [1, 3, 3, 4]
    expected = []
    for a, b in product(permutations("ab"), permutations("bc")):
        order_a = (*a, "c", "d")
        order_b = ("a", *b, "d")
        expected.append(rank_stability((order_a, order_b), depth=1).mean_rank_biased_overlap)
    assert rank_biased_overlap(left, right, tie_method="independent") == pytest.approx(
        np.mean(expected)
    )
    assert rank_biased_overlap([1, 2, 3, 4], [4, 1, 2, 3]) == pytest.approx(
        rank_stability((names, tuple("bcda")), depth=1).mean_rank_biased_overlap
    )
    assert rank_biased_overlap([1] * 4, [1] * 4, tie_method="independent") < 1
    for args in [([1, 2], [1]), ([1, np.nan], [1, 2])]:
        with pytest.raises(ValueError):
            rank_biased_overlap(*args)
    with pytest.raises(ValueError):
        rank_biased_overlap(left, right, persistence=1)


def test_consensus_preserves_evidence_ties_and_excludes_zero_support():
    result = consensus_ranking(tuple("dcba"), [[0, 1, 3, 3], [0, 1, 3, 3]], depth=1)
    assert result.ranking == tuple("abcd")
    assert result.ranks == (3.5, 3.5, 1.5, 1.5)
    assert result.inclusion_probabilities == (0, 0, 0.5, 0.5)
    with pytest.raises(ValueError):
        consensus_ranking(tuple("ab"), [[1, -1], [1, 0]], depth=1)


def test_auc_reverse_km_ties_and_weighted_pair_formula():
    # At t=1 remove the event before censoring: G(1)=1-1/3=2/3.
    # At t=2 censor one of two: G(2)=1/3. Case weights are 1.5 and 3.
    train_t, train_e = [1, 1, 2, 4], [1, 0, 0, 1]
    time, event, score = [1, 2, 4, 5], [1, 1, 0, 0], [0, 2, 1, 3]
    result = cumulative_dynamic_auc(train_t, train_e, time, event, score, horizon=3)
    assert result == pytest.approx(3 / (4.5 * 2))
    # Case at horizon counts; censoring at horizon does not become a control.
    assert cumulative_dynamic_auc([1, 4], [1, 1], [3, 3, 4], [1, 0, 0], [2, 9, 1], horizon=3) == 1


def test_auc_uncensored_matches_sklearn_including_ties():
    time, event = np.array([1.0, 2, 3, 4, 5]), np.ones(5, dtype=bool)
    scores = np.array([0.2, 0.8, 0.8, 0.8, 0.1])
    result = cumulative_dynamic_auc(time, event, time, event, scores, horizon=2.5)
    assert result == pytest.approx(roc_auc_score(time <= 2.5, scores))
    # Adjacent near-ties form a single group, even when endpoints differ > tol.
    assert cumulative_dynamic_auc(
        time, event, time, event, np.arange(5) * 0.9e-8, horizon=2.5, tied_tolerance=1e-8
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="cases and controls"):
        cumulative_dynamic_auc(time, event, time, event, scores, horizon=0.5)
    with pytest.raises(ValueError, match="horizon"):
        cumulative_dynamic_auc(time, event, time, event, scores, horizon=5)
    with pytest.raises(ValueError, match="horizon"):
        cumulative_dynamic_auc([1, 2, 3], [0, 0, 0], [1, 4], [1, 0], [2, 1], horizon=3)


def test_shared_bootstrap_keeps_pairing_and_uses_percentile_interval():
    observed = []
    x = np.arange(12.0)

    def statistic(indices):
        observed.append(indices.copy())
        return float(x[indices].mean()), float((x[indices] + 2).mean())

    result = paired_bootstrap_difference(tuple(map(str, range(12))), statistic, resamples=20)
    assert result.difference == 2
    np.testing.assert_allclose(result.differences, 2)
    assert result.confidence_interval_low == pytest.approx(2)
    replay = paired_bootstrap_difference(tuple(map(str, range(12))), statistic, resamples=20)
    assert replay == result
    for first, second in zip(observed[:21], observed[21:], strict=True):
        np.testing.assert_array_equal(first, second)


def test_bootstrap_explicit_failures_and_statistic_errors():
    def sparse(indices):
        return None if np.unique(indices).size == 1 else (0.0, 1.0)

    result = paired_bootstrap_difference(("a", "b"), sparse, resamples=100)
    assert 0 < result.failed_resamples < 100
    assert sum(x is None for x in result.differences) == result.failed_resamples
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_difference(("a", "b"), lambda i: (np.nan, 1))
    with pytest.raises(ValueError, match="undefined"):
        paired_bootstrap_difference(("a", "b"), lambda i: None)


def test_learnability_gap_has_oracle_minus_fitted_sign():
    labels = [0, 0, 1, 1]

    def metric(risk):
        return roc_auc_score(labels, risk)

    result = learnability_gap(labels, [1, 1, 0, 0], metric=metric)
    assert (result.oracle_auc, result.fitted_auc, result.gap) == (1, 0, 1)
    assert learnability_gap(labels, labels, metric=metric).gap == 0
    assert learnability_gap(labels, [1] * 4, metric=metric).gap == 0.5
    assert learnability_gap([1] * 4, labels, metric=metric).gap == -0.5


def test_webber_whole_tie_prefix_agreement_and_rank_direction():
    # At d=1 prefixes {a,b} and {a}: A=2/3. At d=2 prefixes
    # {a,b} and {a,b,c}: A=4/5; at d=3 both are {a,b,c}.
    p = 0.7
    expected = (1 - p) * (2 / 3 + p * 4 / 5 + p**2) + p**3
    assert rank_biased_overlap([1, 1, 3, 4], [1, 2, 2, 4], persistence=p) == pytest.approx(expected)
    assert rank_biased_overlap([1] * 4, [1] * 4) == pytest.approx(1)
    assert rank_biased_overlap([1, 1, 3, 4], [1, 1, 3, 4]) == pytest.approx(1)
    from scipy.stats import rankdata

    scores = np.array([9.0, 2.0, 1.0, 0.0])
    assert rank_biased_overlap(rankdata(-scores), [1, 2, 3, 4]) == 1
    with pytest.raises(ValueError, match="same shape"):
        rank_biased_overlap([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="persistence"):
        rank_biased_overlap([1, 2], [1, 2], persistence=np.nan)
    with pytest.raises(ValueError, match="tie_method"):
        rank_biased_overlap([1, 2], [1, 2], tie_method="other")


def test_never_selected_median_is_worse_than_wide_boundary_tie():
    result = consensus_ranking(tuple("abcde"), [[1, 1, 1, 1, 0]] * 2, depth=1)
    assert result.inclusion_probabilities == (0.25, 0.25, 0.25, 0.25, 0)
    assert result.median_selected_ranks[:4] == (2.5,) * 4
    assert np.isinf(result.median_selected_ranks[-1])
    assert result.ranking[-1] == "e"
    assert min(result.ranks[-1:]) > max(result.ranks[:4])


def test_exact_auc_ties_do_not_merge_dense_probabilities():
    times = np.arange(1.0, 1002.0)
    events = np.ones(times.size)
    scores = 0.005 - times * 0.9e-8
    assert cumulative_dynamic_auc(times, events, times, events, scores, horizon=500) == 1
    assert cumulative_dynamic_auc(
        times, events, times, events, scores, horizon=500, tied_tolerance=1e-8
    ) == pytest.approx(0.5)
    assert cumulative_dynamic_auc(times, events, times, events, scores * 1e-8, horizon=500) == 1


@pytest.mark.parametrize(
    "change, message",
    [
        ({"risk": [1]}, "aligned"),
        ({"horizon": float("nan")}, "finite and positive"),
        ({"horizon": 5}, "strictly earlier"),
        ({"tied_tolerance": -1}, "non-negative"),
    ],
)
def test_auc_errors_identify_the_invalid_argument(change, message):
    kwargs = dict(
        training_time=[1, 5],
        training_event=[1, 1],
        time=[1, 5],
        event=[1, 1],
        risk=[2, 1],
        horizon=3,
    )
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        cumulative_dynamic_auc(**kwargs)


def test_metrics_discovery_bootstrap_count_and_gap_alias():
    import pyplasmode as ppm
    from pyplasmode.metrics import discrimination_gap

    assert "metrics" in ppm.__all__
    assert ppm.metrics.discrimination_gap is discrimination_gap
    assert learnability_gap is discrimination_gap

    def statistic(i):
        return float(i.mean()), float(i.mean()) + 1

    count = paired_bootstrap_difference(4, statistic, resamples=20)
    ids = paired_bootstrap_difference(["a", "b", "c", "d"], statistic, resamples=20)
    assert count == ids
    assert (
        paired_bootstrap_difference(np.array(["a", "b", "c", "d"]), statistic, resamples=20) == ids
    )
    for array in (np.array([["a", "b"]]), np.array(["a", "a"])):
        with pytest.raises(ValueError):
            paired_bootstrap_difference(array, statistic, resamples=20)
    for participants in (True, 1, "abc", ["a", "a"]):
        with pytest.raises(ValueError):
            paired_bootstrap_difference(participants, statistic, resamples=20)


def test_bootstrap_gap_interval_uses_fitted_then_oracle_order():
    from pyplasmode.metrics import discrimination_gap

    labels = np.array([0, 0, 1, 1, 0, 1])
    oracle = np.array([0.1, 0.2, 0.4, 0.8, 0.5, 0.9])
    fitted = np.array([0.1, 0.3, 0.8, 0.6, 0.2, 0.9])
    gap = discrimination_gap(oracle, fitted, metric=lambda r: roc_auc_score(labels, r))

    def statistic(i):
        if np.unique(labels[i]).size < 2:
            return None
        return roc_auc_score(labels[i], fitted[i]), roc_auc_score(labels[i], oracle[i])

    interval = paired_bootstrap_difference(len(labels), statistic, resamples=100)
    assert interval.difference == gap.gap
    assert interval.failed_resamples > 0


def test_unequal_nested_budgets_cannot_have_perfect_stability():
    names = tuple("abcd")
    nested = [nomination_from_scores(names, row, depth=2) for row in [[1, 0, 0, 0], [1, 1, 0, 0]]]
    assert selection_stability(nested).value < 1
