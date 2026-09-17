"""Empirical populations and deterministic complete-row resampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyplasmode._validation import identities as validate_identities
from pyplasmode._validation import indices as integer_indices
from pyplasmode._validation import integer

type SamplingMethod = Literal["with_replacement", "without_replacement"]


def _readonly_float_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty two-dimensional numeric matrix")
    if np.isinf(matrix).any():
        raise ValueError(f"{name} contains infinite values")
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_indices(values: NDArray[np.int64]) -> NDArray[np.int64]:
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class Population:
    """A numeric feature matrix with stable ordered column identities.

    Missing values are retained. Infinite values, duplicate identities and shape mismatches
    are rejected.
    """

    X: NDArray[np.float64]
    feature_ids: tuple[str, ...]

    def __init__(self, X: ArrayLike, feature_ids: tuple[str, ...]) -> None:
        matrix = _readonly_float_matrix(X, name="X")
        identities = tuple(feature_ids)
        if len(identities) != matrix.shape[1]:
            raise ValueError("feature_ids length must equal the number of X columns")
        validate_identities(identities, "feature_ids")
        object.__setattr__(self, "X", matrix)
        object.__setattr__(self, "feature_ids", identities)

    @property
    def sample_count(self) -> int:
        """Return the number of empirical rows."""
        return int(self.X.shape[0])

    @property
    def feature_count(self) -> int:
        """Return the number of feature columns."""
        return int(self.X.shape[1])


@dataclass(frozen=True, slots=True)
class ResampledPopulation:
    """A complete-row sample and its in-memory source-row indices."""

    X: NDArray[np.float64]
    feature_ids: tuple[str, ...]
    source_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        matrix = _readonly_float_matrix(self.X, name="X")
        indices = integer_indices(self.source_indices, "source_indices")
        validate_identities(self.feature_ids, "feature_ids")
        if len(self.feature_ids) != matrix.shape[1]:
            raise ValueError("feature_ids length must equal the number of X columns")
        if indices.ndim != 1 or indices.shape[0] != matrix.shape[0]:
            raise ValueError("source_indices must contain one index per resampled row")
        object.__setattr__(self, "X", matrix)
        object.__setattr__(self, "source_indices", _readonly_indices(indices))

    def as_population(self) -> Population:
        """Return the sampled rows as a population for truth materialisation."""
        return Population(self.X, self.feature_ids)


@dataclass(frozen=True, slots=True)
class PopulationPartition:
    """Disjoint empirical source populations with original-row lineage."""

    training: Population
    validation: Population
    evaluation: Population
    training_source_indices: NDArray[np.int64]
    validation_source_indices: NDArray[np.int64]
    evaluation_source_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Require one complete, disjoint partition over a common feature universe."""
        populations = (self.training, self.validation, self.evaluation)
        indices = tuple(
            _readonly_indices(integer_indices(values, "partition lineage"))
            for values in (
                self.training_source_indices,
                self.validation_source_indices,
                self.evaluation_source_indices,
            )
        )
        if any(
            values.ndim != 1 or values.size != population.sample_count
            for values, population in zip(indices, populations, strict=True)
        ):
            raise ValueError("partition lineage must identify every source row")
        if len({population.feature_ids for population in populations}) != 1:
            raise ValueError("partition populations must share one feature universe")
        sets = tuple(set(values.tolist()) for values in indices)
        if any(len(unique) != values.size for unique, values in zip(sets, indices, strict=True)):
            raise ValueError("partition source identities must be unique within each role")
        if any(
            left & right for position, left in enumerate(sets) for right in sets[position + 1 :]
        ):
            raise ValueError("source identities cannot cross partition roles")
        combined = set().union(*sets)
        if combined != set(range(len(combined))):
            raise ValueError("partition lineage must be a complete zero-based source universe")
        object.__setattr__(self, "training_source_indices", indices[0])
        object.__setattr__(self, "validation_source_indices", indices[1])
        object.__setattr__(self, "evaluation_source_indices", indices[2])


def partition_population(
    population: Population,
    *,
    training_fraction: float,
    validation_fraction: float,
    seed: int,
) -> PopulationPartition:
    """Partition empirical rows before any complete-row resampling.

    The evaluation fraction is the remainder after training and validation.
    Fractions determine integer boundaries by floor; every role must retain at
    least one source row.
    """
    integer(seed, "seed")
    if (
        not np.isfinite(training_fraction)
        or not np.isfinite(validation_fraction)
        or training_fraction <= 0.0
        or validation_fraction <= 0.0
        or training_fraction + validation_fraction >= 1.0
    ):
        raise ValueError("population partition settings are invalid")
    training_count = int(np.floor(population.sample_count * training_fraction))
    validation_count = int(np.floor(population.sample_count * validation_fraction))
    evaluation_count = population.sample_count - training_count - validation_count
    if min(training_count, validation_count, evaluation_count) < 1:
        raise ValueError("population partition requires at least one source row per role")
    order = np.random.default_rng(seed).permutation(population.sample_count)
    training_indices = np.asarray(order[:training_count], dtype=np.int64)
    validation_indices = np.asarray(
        order[training_count : training_count + validation_count], dtype=np.int64
    )
    evaluation_indices = np.asarray(order[-evaluation_count:], dtype=np.int64)
    return PopulationPartition(
        Population(population.X[training_indices], population.feature_ids),
        Population(population.X[validation_indices], population.feature_ids),
        Population(population.X[evaluation_indices], population.feature_ids),
        training_indices,
        validation_indices,
        evaluation_indices,
    )


def resample_population(
    population: Population,
    sample_size: int,
    rng: np.random.Generator,
) -> ResampledPopulation:
    """Sample complete empirical rows with replacement using ``rng``."""
    integer(sample_size, "sample_size", minimum=1)
    indices = rng.integers(0, population.sample_count, size=sample_size, dtype=np.int64)
    return ResampledPopulation(population.X[indices], population.feature_ids, indices)


def sample_population(
    population: Population,
    sample_size: int,
    rng: np.random.Generator,
    *,
    method: SamplingMethod = "with_replacement",
) -> ResampledPopulation:
    """Sample complete empirical rows under an explicit replacement policy.

    Args:
        population: Source feature matrix and column identities.
        sample_size: Number of rows to return.
        rng: NumPy random generator controlling row selection.
        method: Sampling with replacement or sampling without replacement.

    Returns:
        Selected complete rows and their source indices.

    Raises:
        ValueError: If the size or method is invalid, or a sample without
            replacement exceeds the source population.
    """
    if method == "with_replacement":
        return resample_population(population, sample_size, rng)
    if method != "without_replacement":
        raise ValueError("sampling method is unsupported")
    integer(sample_size, "sample_size", minimum=1)
    if sample_size > population.sample_count:
        raise ValueError("sample_size exceeds the population without replacement")
    indices = np.asarray(
        rng.choice(population.sample_count, size=sample_size, replace=False),
        dtype=np.int64,
    )
    return ResampledPopulation(population.X[indices], population.feature_ids, indices)
