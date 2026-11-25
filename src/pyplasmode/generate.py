"""Simulate outcomes on sampled rows, optionally in separate training, validation and test sets."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from pyplasmode._validation import indices as integer_indices
from pyplasmode._validation import integer
from pyplasmode.data import Population, PopulationPartition, SamplingMethod, sample_population
from pyplasmode.outcomes import (
    BinaryGeneratedOutcome,
    ContinuousGeneratedOutcome,
    CountGeneratedOutcome,
    GeneratedOutcome,
    OrdinalGeneratedOutcome,
    OutcomeSpec,
    SurvivalGeneratedOutcome,
    generate_outcome,
)
from pyplasmode.structures import FeatureSets
from pyplasmode.truth import (
    CorrelatedTruth,
    CustomTruth,
    MaterializedTruth,
    SparseTruth,
    TruthSpec,
    materialize_truth,
)


@dataclass(frozen=True, slots=True)
class Plasmode:
    """Sampled features paired with simulated outcomes.

    Attributes:
        X: Read-only feature matrix in draw order.
        feature_ids: Feature IDs in column order.
        outcome: Simulated outcomes in row order, with model parameters.
        truth: Score driving the outcomes and known feature-selection targets.
        source_indices: Zero-based row indices into this sample's input pool.
        seed: Generation seed for `generate`, or this pool's row-sampling seed for a split dataset.
        sampling_method: Whether source rows were drawn with or without replacement.
    """

    X: NDArray[np.float64]
    feature_ids: tuple[str, ...]
    outcome: GeneratedOutcome
    truth: MaterializedTruth
    source_indices: NDArray[np.int64]
    seed: int
    sampling_method: SamplingMethod


@dataclass(frozen=True, slots=True)
class PartitionSampleSizes:
    """Positive row counts for drawing separate datasets.

    Attributes:
        training: Row count for fitting.
        validation: Row count for tuning.
        evaluation: Row count for testing.
    """

    training: int
    validation: int
    evaluation: int

    def __post_init__(self) -> None:
        """Require a positive integer count for every role."""
        for value in (self.training, self.validation, self.evaluation):
            integer(value, "partition sample size", minimum=1)


@dataclass(frozen=True, slots=True)
class PartitionedPlasmode:
    """Simulated datasets for fitting, tuning and testing drawn from separate source pools.

    Attributes:
        training: Dataset for fitting.
        validation: Dataset for tuning.
        evaluation: Dataset for testing.
        training_source_indices: Original training row indices, zero-based.
        validation_source_indices: Original validation row indices, zero-based.
        evaluation_source_indices: Original test row indices, zero-based.

    Indices can repeat within a dataset. Each `Plasmode.source_indices` refers to its pool.
    """

    training: Plasmode
    validation: Plasmode
    evaluation: Plasmode
    training_source_indices: NDArray[np.int64]
    validation_source_indices: NDArray[np.int64]
    evaluation_source_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate that source-row indices are disjoint across generated roles."""
        samples = (self.training, self.validation, self.evaluation)
        lineages = tuple(
            integer_indices(values, "generated source lineage")
            for values in (
                self.training_source_indices,
                self.validation_source_indices,
                self.evaluation_source_indices,
            )
        )
        if any(
            lineage.ndim != 1 or lineage.size != sample.X.shape[0]
            for lineage, sample in zip(lineages, samples, strict=True)
        ):
            raise ValueError("generated role lineage is incomplete")
        identities = tuple(set(values.tolist()) for values in lineages)
        if any(
            left & right
            for position, left in enumerate(identities)
            for right in identities[position + 1 :]
        ):
            raise ValueError("an empirical source identity crosses generated roles")
        for name, lineage in zip(
            (
                "training_source_indices",
                "validation_source_indices",
                "evaluation_source_indices",
            ),
            lineages,
            strict=True,
        ):
            frozen = np.array(lineage, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)


def generate(
    population: Population,
    *,
    truth: TruthSpec,
    outcome: OutcomeSpec,
    sample_size: int,
    seed: int,
    sampling_method: SamplingMethod = "with_replacement",
    feature_sets: FeatureSets | None = None,
) -> Plasmode:
    """Sample whole feature rows and simulate one outcome per row from a known score.

    Args:
        truth: Specification of the outcome-generating score and feature-selection targets.
        outcome: Outcome family, target summaries and effect or target AUC.
        sample_size: Positive number of feature rows to draw.
        seed: Non-negative seed for sampling rows, choosing features and drawing outcomes.
        sampling_method: Whether repeats are allowed when drawing rows.
        feature_sets: Feature groups required for correlated and module scores.

    Raises:
        ValueError: The score is invalid or sampling without replacement needs more rows.
        CalibrationError: Outcome targets are unattainable.
        RealizationError: The simulated outcomes fail a sampling diagnostic.
    """
    integer(seed, "seed")
    row_seed, truth_seed, outcome_seed = np.random.SeedSequence(seed).spawn(3)
    sampled = sample_population(
        population,
        sample_size,
        np.random.default_rng(row_seed),
        method=sampling_method,
    )
    materialized = materialize_truth(
        truth,
        sampled.as_population(),
        seed=int(truth_seed.generate_state(1, dtype=np.uint32)[0]),
        feature_sets=feature_sets,
    )
    generated = generate_outcome(outcome, materialized.signal, np.random.default_rng(outcome_seed))
    return Plasmode(
        sampled.X,
        sampled.feature_ids,
        generated,
        materialized,
        sampled.source_indices,
        seed,
        sampling_method,
    )


def _freeze_partition_truth(
    truth: TruthSpec,
    population: Population,
    *,
    seed: int,
    feature_sets: FeatureSets | None,
) -> TruthSpec:
    """Freeze training-derived identities before generating other roles."""
    if isinstance(truth, CustomTruth):
        raise ValueError("partitioned generation requires a reusable declarative truth")
    if isinstance(truth, SparseTruth) and truth.count is not None:
        materialized = materialize_truth(
            truth,
            population,
            seed=seed,
            feature_sets=feature_sets,
        )
        return SparseTruth(
            features=materialized.direct_features,
            weights=tuple(item.weight for item in materialized.weights),
            max_absolute_correlation=truth.max_absolute_correlation,
        )
    if (
        isinstance(truth, CorrelatedTruth)
        and truth.signal_form == "sentinels"
        and not truth.sentinels
    ):
        materialized = materialize_truth(
            truth,
            population,
            seed=seed,
            feature_sets=feature_sets,
        )
        return CorrelatedTruth(
            truth.group_names,
            materialized.direct_features,
            weights=truth.weights,
            signal_form=truth.signal_form,
        )
    return truth


def _slice_outcome(outcome: GeneratedOutcome, rows: slice) -> GeneratedOutcome:
    """Return the outcomes for one role from a jointly generated sample."""
    if isinstance(outcome, BinaryGeneratedOutcome):
        return BinaryGeneratedOutcome(outcome.values[rows], outcome.calibration)
    if isinstance(outcome, ContinuousGeneratedOutcome):
        return ContinuousGeneratedOutcome(outcome.values[rows], outcome.calibration)
    if isinstance(outcome, CountGeneratedOutcome):
        return CountGeneratedOutcome(
            outcome.values[rows], outcome.exposure[rows], outcome.calibration
        )
    if isinstance(outcome, OrdinalGeneratedOutcome):
        return OrdinalGeneratedOutcome(outcome.codes[rows], outcome.categories, outcome.calibration)
    if isinstance(outcome, SurvivalGeneratedOutcome):
        return SurvivalGeneratedOutcome(
            outcome.time[rows],
            outcome.event[rows],
            outcome.latent_event_time[rows],
            outcome.censoring_time[rows],
            outcome.calibration,
        )
    raise TypeError(f"unsupported generated outcome: {type(outcome).__name__}")


def generate_partitioned(
    partition: PopulationPartition,
    *,
    truth: TruthSpec,
    outcome: OutcomeSpec,
    sample_sizes: PartitionSampleSizes,
    seed: int,
    sampling_method: SamplingMethod = "with_replacement",
    feature_sets: FeatureSets | None = None,
) -> PartitionedPlasmode:
    """Simulate outcomes on samples drawn from separate training, validation and test pools.

    Automatic feature choices use the training pool. Preprocessing, score scaling and
    outcome calibration use the pooled sampled rows. A role's expected rate can
    differ from the combined-population target with its covariate distribution,
    and its realised rate also varies with outcome sampling.

    Args:
        truth: Built-in specification of the outcome-generating score and selection targets.
        outcome: Outcome targets for the combined sample.
        seed: Non-negative seed for rows, features and outcomes.
        sampling_method: Whether to draw the rows in each pool with replacement.
        feature_sets: Feature groups required for correlated and module scores.

    Raises:
        ValueError: The score is invalid or custom, or a draw without replacement exceeds its pool.
        CalibrationError: Outcome targets are unattainable.
        RealizationError: The combined outcomes fail a sampling diagnostic.
    """
    integer(seed, "seed")
    role_seeds = np.random.SeedSequence(seed).spawn(6)
    (
        truth_selection_seed,
        training_seed,
        validation_seed,
        evaluation_seed,
        truth_materialization_seed,
        outcome_seed,
    ) = role_seeds
    frozen_truth = _freeze_partition_truth(
        truth,
        partition.training,
        seed=int(truth_selection_seed.generate_state(1, dtype=np.uint32)[0]),
        feature_sets=feature_sets,
    )
    role_inputs = (
        (
            partition.training,
            partition.training_source_indices,
            sample_sizes.training,
            training_seed,
        ),
        (
            partition.validation,
            partition.validation_source_indices,
            sample_sizes.validation,
            validation_seed,
        ),
        (
            partition.evaluation,
            partition.evaluation_source_indices,
            sample_sizes.evaluation,
            evaluation_seed,
        ),
    )
    sampled = []
    original_lineage: list[NDArray[np.int64]] = []
    for population, source_indices, sample_size, role_seed in role_inputs:
        role_seed_value = int(role_seed.generate_state(1, dtype=np.uint32)[0])
        sample = sample_population(
            population,
            sample_size,
            np.random.default_rng(role_seed_value),
            method=sampling_method,
        )
        sampled.append((sample, role_seed_value))
        original_lineage.append(source_indices[sample.source_indices])
    combined = Population(
        np.vstack(tuple(sample.X for sample, _ in sampled)),
        partition.training.feature_ids,
    )
    materialized = materialize_truth(
        frozen_truth,
        combined,
        seed=int(truth_materialization_seed.generate_state(1, dtype=np.uint32)[0]),
        feature_sets=feature_sets,
    )
    generated_outcome = generate_outcome(
        outcome,
        materialized.signal,
        np.random.default_rng(outcome_seed),
    )
    generated: list[Plasmode] = []
    start = 0
    for sample, role_seed_value in sampled:
        stop = start + sample.X.shape[0]
        rows = slice(start, stop)
        generated.append(
            Plasmode(
                sample.X,
                sample.feature_ids,
                _slice_outcome(generated_outcome, rows),
                replace(materialized, signal=materialized.signal[rows]),
                sample.source_indices,
                role_seed_value,
                sampling_method,
            )
        )
        start = stop
    return PartitionedPlasmode(
        generated[0],
        generated[1],
        generated[2],
        original_lineage[0],
        original_lineage[1],
        original_lineage[2],
    )
