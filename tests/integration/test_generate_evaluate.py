"""Public generation-to-evaluation composition."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pyplasmode import (
    BinaryGeneratedOutcome,
    BinaryOutcome,
    CorrelatedTruth,
    FeatureSet,
    FeatureSets,
    PartitionSampleSizes,
    Population,
    SparseTruth,
    evaluate_prediction,
    evaluate_ranking,
    generate,
    generate_partitioned,
    partition_population,
)


@pytest.mark.integration
def test_generate_fit_rank_and_evaluate_without_model_ownership() -> None:
    rng = np.random.default_rng(1)
    population = Population(rng.normal(size=(2_000, 20)), tuple(f"p{i}" for i in range(20)))
    sample = generate(
        population,
        truth=SparseTruth(features=("p0", "p1")),
        outcome=BinaryOutcome(0.2, odds_ratio=3.0),
        sample_size=5_000,
        seed=7,
    )
    assert isinstance(sample.outcome, BinaryGeneratedOutcome)
    model = LogisticRegression(C=1.0, max_iter=1_000).fit(sample.X, sample.outcome.values)
    ranking = tuple(sample.feature_ids[index] for index in np.argsort(-np.abs(model.coef_[0])))

    recovery = evaluate_ranking(ranking, sample.truth, depths=(2, 5))
    prediction = evaluate_prediction(
        sample.outcome.values, model.predict_proba(sample.X)[:, 1], metric="roc_auc"
    )

    assert recovery.at_depth[0].recall == 1.0
    assert prediction.value > 0.7


@pytest.mark.integration
def test_partitioned_generation_freezes_truth_and_prevents_source_role_leakage() -> None:
    """Data-derived sparse truth is fixed on training and lineage remains disjoint."""
    rng = np.random.default_rng(21)
    population = Population(rng.normal(size=(600, 30)), tuple(f"p{i}" for i in range(30)))
    partition = partition_population(
        population,
        training_fraction=0.7,
        validation_fraction=0.15,
        seed=23,
    )

    sample = generate_partitioned(
        partition,
        truth=SparseTruth(count=3),
        outcome=BinaryOutcome(0.15, odds_ratio=2.0),
        sample_sizes=PartitionSampleSizes(1_000, 300, 300),
        seed=29,
    )

    assert sample.training.truth.direct_features == sample.validation.truth.direct_features
    assert sample.training.truth.direct_features == sample.evaluation.truth.direct_features
    roles = (
        set(sample.training_source_indices.tolist()),
        set(sample.validation_source_indices.tolist()),
        set(sample.evaluation_source_indices.tolist()),
    )
    assert not roles[0] & roles[1]
    assert not roles[0] & roles[2]
    assert not roles[1] & roles[2]


@pytest.mark.integration
def test_partitioned_generation_can_preserve_every_fixed_role_member_once() -> None:
    """Cohort-matched generation can retain complete disjoint empirical roles."""
    rng = np.random.default_rng(61)
    population = Population(rng.normal(size=(100, 8)), tuple(f"p{i}" for i in range(8)))
    partition = partition_population(
        population,
        training_fraction=0.70,
        validation_fraction=0.15,
        seed=67,
    )

    sample = generate_partitioned(
        partition,
        truth=SparseTruth(features=("p0", "p1")),
        outcome=BinaryOutcome(0.20, odds_ratio=2.0),
        sample_sizes=PartitionSampleSizes(70, 15, 15),
        seed=71,
        sampling_method="without_replacement",
    )

    assert len(set(sample.training_source_indices.tolist())) == 70
    assert len(set(sample.validation_source_indices.tolist())) == 15
    assert len(set(sample.evaluation_source_indices.tolist())) == 15
    combined_signal = np.concatenate(
        (
            sample.training.truth.signal,
            sample.validation.truth.signal,
            sample.evaluation.truth.signal,
        )
    )
    np.testing.assert_allclose(np.mean(combined_signal), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(combined_signal), 1.0, atol=1e-12)
    assert sample.training.outcome.calibration == sample.validation.outcome.calibration
    assert sample.training.outcome.calibration == sample.evaluation.outcome.calibration


@pytest.mark.integration
def test_partitioned_generation_tolerates_role_local_missing_non_signal() -> None:
    """A wholly unavailable irrelevant column cannot invalidate another role."""
    rng = np.random.default_rng(31)
    values = rng.normal(size=(600, 30))
    feature_ids = tuple(f"p{i}" for i in range(30))
    initial = partition_population(
        Population(values, feature_ids),
        training_fraction=0.7,
        validation_fraction=0.15,
        seed=37,
    )
    values[initial.validation_source_indices, -1] = np.nan
    partition = partition_population(
        Population(values, feature_ids),
        training_fraction=0.7,
        validation_fraction=0.15,
        seed=37,
    )

    sample = generate_partitioned(
        partition,
        truth=SparseTruth(features=("p0", "p1")),
        outcome=BinaryOutcome(0.15, odds_ratio=2.0),
        sample_sizes=PartitionSampleSizes(1_000, 300, 300),
        seed=41,
    )

    assert np.isnan(sample.validation.X[:, -1]).all()
    assert sample.validation.truth.direct_features == ("p0", "p1")
    assert np.isfinite(sample.validation.truth.signal).all()


@pytest.mark.integration
def test_partitioned_generation_preserves_correlated_signal_contract() -> None:
    """Training-derived sentinels cannot discard weights or the group-mean signal form."""
    rng = np.random.default_rng(43)
    population = Population(rng.normal(size=(600, 8)), tuple(f"p{i}" for i in range(8)))
    partition = partition_population(
        population,
        training_fraction=0.7,
        validation_fraction=0.15,
        seed=47,
    )
    feature_sets = FeatureSets(
        (
            FeatureSet("g1", ("p0", "p1")),
            FeatureSet("g2", ("p2", "p3", "p4")),
        )
    )

    sample = generate_partitioned(
        partition,
        truth=CorrelatedTruth(
            ("g1", "g2"),
            weights=(2.0, -0.5),
            signal_form="group_means",
        ),
        outcome=BinaryOutcome(0.2, odds_ratio=2.0),
        sample_sizes=PartitionSampleSizes(700, 200, 200),
        seed=53,
        feature_sets=feature_sets,
    )

    assert sample.training.truth.description == "Correlated group_means truth over 2 groups"
    assert sample.validation.truth.description == "Correlated group_means truth over 2 groups"
    assert tuple(item.weight for item in sample.evaluation.truth.weights) == (
        1.0,
        1.0,
        -0.5 / 3,
        -0.5 / 3,
        -0.5 / 3,
    )
    assert sample.training.truth.direct_features == ()
    assert sample.training.truth.direct_features == sample.validation.truth.direct_features
