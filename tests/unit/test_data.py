"""Population and empirical resampling contracts."""

import numpy as np
import pytest

from pyplasmode import Population, partition_population, resample_population, sample_population


@pytest.mark.unit
def test_population_copies_validates_and_freezes_input() -> None:
    source = np.array([[1.0, np.nan], [2.0, 3.0]])
    population = Population(source, ("a", "b"))
    source[0, 0] = 99.0

    assert population.X[0, 0] == 1.0
    assert not population.X.flags.writeable
    with pytest.raises(ValueError, match="unique"):
        Population(np.ones((2, 2)), ("a", "a"))
    with pytest.raises(ValueError, match="infinite"):
        Population(np.array([[1.0, np.inf]]), ("a", "b"))


@pytest.mark.unit
def test_seeded_resampling_returns_complete_source_rows() -> None:
    source = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 5.0]])
    population = Population(source, ("a", "b"))
    first = resample_population(population, 20, np.random.default_rng(7))
    second = resample_population(population, 20, np.random.default_rng(7))

    np.testing.assert_array_equal(first.source_indices, second.source_indices)
    np.testing.assert_array_equal(first.X, source[first.source_indices])
    assert np.isnan(first.X[:, 1]).sum() == np.sum(first.source_indices == 0)
    with pytest.raises(ValueError, match="sample_size"):
        resample_population(population, 0, np.random.default_rng(7))


def test_sampling_without_replacement_returns_unique_complete_source_rows() -> None:
    """The explicit cohort-preserving policy never duplicates a source row."""
    source = np.arange(30, dtype=np.float64).reshape(10, 3)
    population = Population(source, ("a", "b", "c"))

    sample = sample_population(
        population,
        10,
        np.random.default_rng(13),
        method="without_replacement",
    )

    assert len(set(sample.source_indices.tolist())) == 10
    np.testing.assert_array_equal(sample.X, source[sample.source_indices])
    with pytest.raises(ValueError, match="exceeds"):
        sample_population(
            population,
            11,
            np.random.default_rng(13),
            method="without_replacement",
        )


def test_population_partition_is_complete_disjoint_and_deterministic() -> None:
    """Source roles are assigned once before any row is resampled."""
    population = Population(
        np.arange(120, dtype=np.float64).reshape(40, 3),
        ("a", "b", "c"),
    )

    first = partition_population(
        population,
        training_fraction=0.7,
        validation_fraction=0.15,
        seed=17,
    )
    second = partition_population(
        population,
        training_fraction=0.7,
        validation_fraction=0.15,
        seed=17,
    )

    np.testing.assert_array_equal(first.training_source_indices, second.training_source_indices)
    roles = (
        set(first.training_source_indices.tolist()),
        set(first.validation_source_indices.tolist()),
        set(first.evaluation_source_indices.tolist()),
    )
    assert not roles[0] & roles[1]
    assert not roles[0] & roles[2]
    assert not roles[1] & roles[2]
    assert set().union(*roles) == set(range(40))
