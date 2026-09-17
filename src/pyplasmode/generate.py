"""Composition of empirical resampling, truth materialisation and outcome generation."""

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
    """A generated sample with its feature data, outcomes and known signal."""

    X: NDArray[np.float64]
    feature_ids: tuple[str, ...]
    outcome: GeneratedOutcome
    truth: MaterializedTruth
    source_indices: NDArray[np.int64]
    seed: int
    sampling_method: SamplingMethod


@dataclass(frozen=True, slots=True)
class PartitionSampleSizes:
    """Requested sampled row counts for the three source roles."""

    training: int
    validation: int
    evaluation: int

    def __post_init__(self) -> None:
        """Require a positive integer count for every role."""
        for value in (self.training, self.validation, self.evaluation):
            integer(value, "partition sample size", minimum=1)


@dataclass(frozen=True, slots=True)
class PartitionedPlasmode:
    """Three generated roles with original empirical source-row lineage."""

    training: Plasmode
    validation: Plasmode
    evaluation: Plasmode
    training_source_indices: NDArray[np.int64]
    validation_source_indices: NDArray[np.int64]
    evaluation_source_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Prove that no original source identity crosses generated roles."""
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
    """Resample empirical rows, construct truth and generate a calibrated outcome.

    Args:
        population: Source matrix of shape (observations, features).
        truth: Mechanism defining latent signal and recovery targets.
        outcome: Observable outcome targets and effect per signal standard deviation.
        sample_size: Positive number of generated rows.
        seed: Non-negative seed; separate child streams control rows, truth and outcome.
        sampling_method: With replacement by default; without replacement cannot
            exceed the source population size.
        feature_sets: Named groups or modules used by the truth specification.

    Returns:
        An in-memory plasmode with read-only arrays and local source-row indices.

    Raises:
        ValueError: Invalid settings or an infeasible truth or sampling request.
        CalibrationError: Outcome targets cannot be represented by the selected model.
        RealizationError: A sampled outcome exceeds its diagnostic tolerance.
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
    """Return one role from an outcome generated under a shared mechanism."""
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
    """Generate training, validation and test samples from separate source rows.

    Args:
        partition: Prior source partition from ``partition_population``.
        truth: Transferable built-in mechanism; custom callables are unsupported.
        outcome: Observable outcome targets over the combined generated population.
        sample_sizes: Positive training, validation and evaluation counts.
        seed: Non-negative root seed for role sampling and mechanism generation.
        sampling_method: Explicit with- or without-replacement policy within each role.
        feature_sets: Required named groups or modules.

    Returns:
        Three plasmodes plus original source identities, disjoint across roles.
        Truth identities are frozen on training source rows; sampled roles share
        one signal scaling and outcome calibration. A role's realised rate can
        differ from the combined-population target through sampling variation.

    Raises:
        ValueError: Invalid roles, unsupported custom truth, or infeasible generation.
        CalibrationError: Outcome targets cannot be represented.
        RealizationError: The combined generated outcome fails a diagnostic check.
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
