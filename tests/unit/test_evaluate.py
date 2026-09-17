"""Analytic biomarker evaluation fixtures."""

import numpy as np
import pytest

from pyplasmode import (
    CorrelatedTruth,
    FeatureEdge,
    FeatureGraph,
    FeatureSet,
    FeatureSets,
    ModuleTruth,
    Population,
    SparseTruth,
    evaluate_graph_region,
    evaluate_groups,
    evaluate_module,
    evaluate_panel,
    evaluate_prediction,
    evaluate_ranking,
    evaluate_signal_reconstruction,
    materialize_truth,
    rank_stability,
    summarize_replicates,
)


def _population() -> Population:
    rng = np.random.default_rng(1)
    return Population(rng.normal(size=(100, 6)), tuple("abcdef"))


@pytest.mark.unit
def test_exact_recovery_matches_hand_calculated_ranks() -> None:
    truth = materialize_truth(SparseTruth(features=("a", "c")), _population(), seed=1)
    result = evaluate_ranking(("a", "b", "c", "d", "e", "f"), truth, depths=(2, 4))

    assert result.first_relevant_rank == 1
    assert result.at_depth[0].recall == 0.5
    assert result.at_depth[0].precision == 0.5
    assert result.at_depth[0].false_discovery_fraction == 0.5
    assert result.at_depth[1].recall == 1.0


@pytest.mark.unit
def test_incomplete_unknown_or_duplicate_rankings_fail() -> None:
    truth = materialize_truth(SparseTruth(features=("a",)), _population(), seed=1)
    with pytest.raises(ValueError, match="complete permutation"):
        evaluate_ranking(("a", "b"), truth, depths=(1,))
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_ranking(("a", "a", "c", "d", "e", "f"), truth, depths=(1,))
    with pytest.raises(ValueError, match="unknown"):
        evaluate_ranking(("a", "b", "c", "d", "e", "z"), truth, depths=(1,))


@pytest.mark.unit
def test_group_and_module_recovery_do_not_relabel_substitutes_as_exact() -> None:
    population = _population()
    sets = FeatureSets(
        (
            FeatureSet("g1", ("a", "b")),
            FeatureSet("g2", ("c", "d")),
            FeatureSet("m", ("a", "c", "e")),
        )
    )
    correlated = materialize_truth(
        CorrelatedTruth(("g1", "g2"), sentinels=("a", "c")),
        population,
        feature_sets=sets,
        seed=1,
    )
    module = materialize_truth(
        ModuleTruth("m", weights=(1.0, 2.0, 1.0)), population, feature_sets=sets, seed=1
    )
    ranking = ("b", "d", "f", "e", "a", "c")

    exact = evaluate_ranking(ranking, correlated, depths=(2,))
    groups = evaluate_groups(ranking, correlated, depths=(2,))
    weighted = evaluate_module(ranking, module, depths=(2, 4))

    assert exact.at_depth[0].recall == 0.0
    assert groups.at_depth[0].group_recall == 1.0
    assert weighted.at_depth[0].weighted_coverage == 0.0
    assert weighted.at_depth[1].weighted_coverage == 0.25


@pytest.mark.unit
def test_graph_region_and_latent_reconstruction_match_direct_fixtures() -> None:
    population = _population()
    graph = FeatureGraph(
        tuple("abcdef"),
        (FeatureEdge("a", "b"), FeatureEdge("b", "c"), FeatureEdge("d", "e")),
    )
    graph_result = evaluate_graph_region(
        ("a", "b", "f", "c", "d", "e"), graph, region=("a", "b", "c"), depths=(2,)
    )
    train = np.arange(0, 70)
    test = np.arange(70, 100)
    signal = 2.0 * population.X[:, 0] - population.X[:, 1]
    reconstruction = evaluate_signal_reconstruction(
        population.X,
        population.feature_ids,
        selected_features=("a", "b"),
        signal=signal,
        development_indices=train,
        evaluation_indices=test,
    )

    assert graph_result.at_depth[0].node_coverage == pytest.approx(2 / 3)
    assert graph_result.at_depth[0].edge_coverage == pytest.approx(1 / 2)
    assert reconstruction.r_squared == pytest.approx(1.0)


@pytest.mark.unit
def test_signal_reconstruction_imputes_from_development_rows() -> None:
    """Empirical missingness is completed without observing evaluation values."""
    matrix = np.asarray([[0.0], [2.0], [np.nan], [4.0], [np.nan], [6.0]])
    signal = np.asarray([0.0, 2.0, 1.0, 4.0, 1.0, 6.0])

    result = evaluate_signal_reconstruction(
        matrix,
        ("protein",),
        selected_features=("protein",),
        signal=signal,
        development_indices=np.asarray([0, 1, 2]),
        evaluation_indices=np.asarray([3, 4, 5]),
    )

    assert result.evaluation_count == 3
    assert np.isfinite(result.r_squared)


@pytest.mark.unit
def test_signal_reconstruction_supports_an_empty_panel_baseline() -> None:
    """No selected biomarker is evaluated through a maintained intercept-only model."""
    population = _population()
    signal = population.X[:, 0]

    result = evaluate_signal_reconstruction(
        population.X,
        population.feature_ids,
        selected_features=(),
        signal=signal,
        development_indices=np.arange(0, 70),
        evaluation_indices=np.arange(70, 100),
    )

    assert result.selected_feature_count == 0
    assert result.evaluation_count == 30
    assert np.isfinite(result.r_squared)


@pytest.mark.unit
def test_rank_stability_has_explicit_identical_and_discordant_limits() -> None:
    identical = rank_stability((("a", "b", "c", "d"), ("a", "b", "c", "d")), depth=2)
    discordant = rank_stability((("a", "b", "c", "d"), ("d", "c", "b", "a")), depth=2)

    assert identical.mean_leading_jaccard == 1.0
    assert identical.mean_rank_biased_overlap == 1.0
    assert discordant.mean_leading_jaccard == 0.0
    assert discordant.mean_rank_biased_overlap < 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "target,prediction,metric",
    [
        ([0, 0, 0], [0.1, 0.2, 0.3], "roc_auc"),
        ([0, 0, 0], [0.1, 0.2, 0.3], "average_precision"),
        ([1], [1], "r2"),
        ([1, 1], [1, 2], "r2"),
        ([], [], "rmse"),
        ([0, np.nan], [0.1, 0.2], "roc_auc"),
        ([0, 1], [-0.1, 1.2], "log_loss"),
    ],
)
def test_prediction_rejects_undefined_or_invalid_metrics(target, prediction, metric) -> None:
    with pytest.raises(ValueError):
        evaluate_prediction(target, prediction, metric=metric)


@pytest.mark.unit
def test_prediction_and_panel_preserve_standard_metric_values() -> None:
    assert evaluate_prediction([0, 1], [0.1, 0.9], metric="roc_auc").value == 1.0
    assert evaluate_prediction([0, 0], [0.1, 0.1], metric="log_loss").value == pytest.approx(
        -np.log(0.9)
    )
    panel = evaluate_panel([0, 1], [0.1, 0.9], [0.9, 0.1], metric="roc_auc")
    assert panel.difference == -1.0
    with pytest.raises(ValueError):
        evaluate_panel([0, 0], [0.1, 0.9], [0.9, 0.1], metric="roc_auc")


@pytest.mark.unit
def test_replicate_uncertainty_requires_replication_and_matches_t_reference() -> None:
    from scipy.stats import sem, t

    with pytest.raises(ValueError, match="two"):
        summarize_replicates([0.8])
    with pytest.raises(ValueError, match="integer"):
        summarize_replicates([0.7, 0.8], failure_count=0.5)
    values = np.asarray([0.6, 0.7, 0.8, 0.9])
    result = summarize_replicates(values)
    low, high = t.interval(0.95, len(values) - 1, loc=values.mean(), scale=sem(values))
    assert result.confidence_interval_low == pytest.approx(low)
    assert result.confidence_interval_high == pytest.approx(high)
