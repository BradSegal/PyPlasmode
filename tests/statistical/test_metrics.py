"""Statistical reference checks on synthetic participants and features."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pyplasmode import nomination_from_scores
from pyplasmode.metrics import (
    cumulative_dynamic_auc,
    paired_bootstrap_difference,
    selection_stability,
)

pytestmark = pytest.mark.statistical


def test_exchangeable_unequal_size_selections_are_chance_level():
    rng = np.random.default_rng(503)
    names = tuple(map(str, range(60)))
    estimates = []
    for _ in range(250):
        nominations = []
        for _ in range(15):
            row = np.zeros(60)
            row[rng.choice(60, size=rng.integers(1, 12), replace=False)] = 1
            nominations.append(nomination_from_scores(names, row, depth=12))
        estimates.append(selection_stability(nominations).value)
    assert abs(np.mean(estimates)) < 0.02


def test_ipcw_recovers_latent_auc_under_independent_censoring():
    rng = np.random.default_rng(307)
    size = 60_000
    score = rng.normal(size=size)
    latent = rng.exponential(size=size) / np.exp(score * 0.8)
    censor = rng.exponential(scale=2, size=size)
    times = np.minimum(latent, censor)
    event = latent <= censor
    expected = roc_auc_score(latent <= 1, score)
    actual = cumulative_dynamic_auc(
        times[:30_000], event[:30_000], times[30_000:], event[30_000:], score[30_000:], horizon=1
    )
    assert abs(actual - expected) < 0.015


def test_paired_bootstrap_matches_mean_difference_uncertainty():
    rng = np.random.default_rng(18)
    left = rng.normal(size=1000)
    right = left + rng.normal(0.2, 0.5, size=1000)
    result = paired_bootstrap_difference(
        tuple(map(str, range(1000))),
        lambda i: (float(left[i].mean()), float(right[i].mean())),
        resamples=1200,
        seed=8,
    )
    differences = right - left
    se = differences.std(ddof=1) / np.sqrt(1000)
    assert result.difference == pytest.approx(differences.mean())
    assert result.confidence_interval_low == pytest.approx(
        differences.mean() - 1.96 * se, abs=0.006
    )
    assert result.confidence_interval_high == pytest.approx(
        differences.mean() + 1.96 * se, abs=0.006
    )


def test_random_rank_rbo_matches_exchangeable_prefix_expectation():
    from pyplasmode.metrics import rank_biased_overlap

    rng = np.random.default_rng(92)
    size, persistence = 30, 0.9
    values = [rank_biased_overlap(np.arange(size), rng.permutation(size)) for _ in range(400)]
    expected = (1 - persistence) * sum(
        (depth / size) * persistence ** (depth - 1) for depth in range(1, size)
    ) + persistence ** (size - 1)
    assert np.mean(values) == pytest.approx(expected, abs=0.015)


def test_consensus_recovers_recurrent_signal_from_noisy_refits():
    from pyplasmode.metrics import consensus_ranking

    rng = np.random.default_rng(317)
    scores = np.exp(rng.normal(size=(80, 30)))
    scores[:, :2] *= np.exp(3)
    consensus = consensus_ranking(tuple(f"f{i}" for i in range(30)), scores, depth=2)
    assert set(consensus.ranking[:2]) == {"f0", "f1"}
    assert min(consensus.inclusion_probabilities[:2]) > max(consensus.inclusion_probabilities[2:])


def test_oracle_has_smaller_population_discrimination_loss_than_noisy_score():
    from scipy.special import expit

    from pyplasmode.metrics import learnability_gap

    rng = np.random.default_rng(83)
    oracle = rng.normal(size=30_000)
    labels = rng.binomial(1, expit(oracle))
    fitted = oracle + rng.normal(size=oracle.size)
    gap = learnability_gap(oracle, fitted, metric=lambda score: roc_auc_score(labels, score))
    assert 0.02 < gap.gap < 0.15
