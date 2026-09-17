"""Every outcome composes with disjoint roles and zero-signal controls."""

import numpy as np
import pytest

import pyplasmode as ppm

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("null", [False, True])
@pytest.mark.parametrize("family", ["binary", "continuous", "count", "ordinal", "survival"])
def test_partitioned_outcome_families_share_calibration_and_lineage(family, null) -> None:
    population = ppm.Population(np.random.default_rng(1).normal(size=(1_000, 4)), tuple("abcd"))
    outcomes = {
        "binary": ppm.BinaryOutcome(0.2, odds_ratio=1 if null else 2),
        "continuous": ppm.ContinuousOutcome(10, 2, variance_explained=0 if null else 0.25),
        "count": ppm.CountOutcome(2, rate_ratio=1 if null else 1.5),
        "ordinal": ppm.OrdinalOutcome(("low", "high"), (0.4, 0.6), 1 if null else 1.5),
        "survival": ppm.TimeToEventOutcome({ppm.years(5): 0.2}, hazard_ratio=1 if null else 2),
    }
    partition = ppm.partition_population(
        population, training_fraction=0.6, validation_fraction=0.2, seed=2
    )
    result = ppm.generate_partitioned(
        partition,
        truth=ppm.NullTruth() if null else ppm.SparseTruth(features=("a",)),
        outcome=outcomes[family],
        sample_sizes=ppm.PartitionSampleSizes(600, 200, 200),
        seed=3,
        sampling_method="without_replacement",
    )
    assert result.training.outcome.calibration == result.evaluation.outcome.calibration
    assert result.training.X.shape == (600, 4)
    assert len(set(result.training_source_indices) & set(result.evaluation_source_indices)) == 0
    for sample in (result.training, result.validation, result.evaluation):
        outcome = sample.outcome
        values = (
            outcome.time
            if isinstance(outcome, ppm.SurvivalGeneratedOutcome)
            else outcome.codes
            if isinstance(outcome, ppm.OrdinalGeneratedOutcome)
            else outcome.values
        )
        assert values.shape == (sample.X.shape[0],)
        assert not values.flags.writeable


def test_single_row_role_is_a_valid_projection_of_a_nonconstant_mechanism() -> None:
    population = ppm.Population(np.arange(40.0).reshape(10, 4), tuple("abcd"))
    partition = ppm.partition_population(
        population, training_fraction=0.8, validation_fraction=0.1, seed=2
    )
    result = ppm.generate_partitioned(
        partition,
        truth=ppm.SparseTruth(features=("a",)),
        outcome=ppm.BinaryOutcome(0.3, odds_ratio=2),
        sample_sizes=ppm.PartitionSampleSizes(8, 1, 1),
        seed=3,
        sampling_method="without_replacement",
    )
    assert result.evaluation.truth.signal.shape == (1,)
