"""Statistical controls for ranking, prediction and panel evaluation."""

import numpy as np
import pytest

from pyplasmode import (
    Population,
    SparseTruth,
    evaluate_panel,
    evaluate_prediction,
    evaluate_ranking,
    materialize_truth,
    summarize_replicates,
)


@pytest.mark.statistical
def test_random_ranking_recall_matches_hypergeometric_expectation() -> None:
    population = Population(
        np.random.default_rng(1).normal(size=(100, 100)),
        tuple(f"f{i}" for i in range(100)),
    )
    truth = materialize_truth(
        SparseTruth(features=("f0", "f1", "f2", "f3", "f4")), population, seed=1
    )
    recalls = []
    rng = np.random.default_rng(4)
    for _ in range(1_000):
        ranking = tuple(rng.permutation(population.feature_ids))
        recalls.append(evaluate_ranking(ranking, truth, depths=(20,)).at_depth[0].recall)
    summary = summarize_replicates(recalls)

    assert abs(summary.mean - 0.2) < 3.0 * summary.standard_error


@pytest.mark.statistical
def test_prediction_and_panel_metrics_delegate_to_standard_quantities() -> None:
    target = np.array([0, 0, 1, 1])
    reference = np.array([0.1, 0.2, 0.8, 0.9])
    panel = np.array([0.1, 0.4, 0.6, 0.8])

    prediction = evaluate_prediction(target, reference, metric="roc_auc")
    retention = evaluate_panel(target, reference, panel, metric="roc_auc")

    assert prediction.value == 1.0
    assert retention.reference_value == 1.0
    assert retention.panel_value == 1.0
    assert retention.difference == 0.0
