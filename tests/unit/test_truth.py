"""Truth materialisation and recovery-semantics contracts."""

import numpy as np
import pytest

from pyplasmode import (
    CorrelatedTruth,
    CustomSignal,
    CustomTruth,
    FeatureSet,
    FeatureSets,
    InteractionTruth,
    ModuleTruth,
    NullTruth,
    Population,
    SparseTruth,
    materialize_truth,
)


def _population() -> Population:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 6))
    X[0, 0] = np.nan
    return Population(X, tuple("abcdef"))


@pytest.mark.unit
def test_truth_forms_keep_exact_group_and_weighted_targets_distinct() -> None:
    population = _population()
    structures = FeatureSets(
        (
            FeatureSet("g1", ("a", "b")),
            FeatureSet("g2", ("c", "d")),
            FeatureSet("module", ("a", "c", "e")),
        )
    )

    null = materialize_truth(NullTruth(), population, seed=1)
    sparse = materialize_truth(SparseTruth(features=("a", "c")), population, seed=1)
    correlated = materialize_truth(
        CorrelatedTruth(group_names=("g1", "g2"), sentinels=("a", "c")),
        population,
        feature_sets=structures,
        seed=1,
    )
    module = materialize_truth(
        ModuleTruth("module", weights=(1.0, 2.0, 1.0)),
        population,
        feature_sets=structures,
        seed=1,
    )

    assert np.all(null.signal == 0.0)
    assert sparse.direct_features == ("a", "c")
    assert correlated.direct_features == ("a", "c")
    assert correlated.admissible_groups[0].features == ("a", "b")
    assert module.direct_features == ()
    assert tuple(item.feature for item in module.weights) == ("a", "c", "e")
    assert np.isclose(np.mean(module.signal), 0.0, atol=1e-12)
    assert np.isclose(np.std(module.signal), 1.0, atol=1e-12)


@pytest.mark.unit
def test_sparse_count_and_interaction_are_seeded_and_nonconstant() -> None:
    population = _population()
    first = materialize_truth(SparseTruth(count=3), population, seed=91)
    second = materialize_truth(SparseTruth(count=3), population, seed=91)
    interaction = materialize_truth(
        InteractionTruth(("a", "b"), operation="product"), population, seed=2
    )

    assert first.direct_features == second.direct_features
    np.testing.assert_allclose(first.signal, second.signal)
    assert np.std(interaction.signal) > 0.0


@pytest.mark.unit
def test_irrelevant_entirely_missing_feature_does_not_block_truth() -> None:
    """A role-local unavailable non-signal feature is constant and ineligible."""
    population = _population()
    values = np.array(population.X, copy=True)
    values[:, -1] = np.nan
    population = Population(values, population.feature_ids)

    explicit = materialize_truth(SparseTruth(features=("a", "c")), population, seed=3)
    selected = materialize_truth(SparseTruth(count=3), population, seed=3)

    assert np.isfinite(explicit.signal).all()
    assert "f" not in selected.direct_features


@pytest.mark.unit
def test_entirely_missing_truth_feature_fails_loudly() -> None:
    """An explicitly required signal cannot be synthesized from absent values."""
    population = _population()
    values = np.array(population.X, copy=True)
    values[:, -1] = np.nan
    population = Population(values, population.feature_ids)

    with pytest.raises(ValueError, match="truth features are entirely missing: \\('f',\\)"):
        materialize_truth(SparseTruth(features=("f",)), population, seed=3)


@pytest.mark.unit
def test_custom_truth_checks_determinism_shape_and_input_mutation() -> None:
    population = _population()

    def valid(
        X: np.ndarray, feature_ids: tuple[str, ...], rng: np.random.Generator
    ) -> CustomSignal:
        first = np.where(np.isnan(X[:, 0]), np.nanmedian(X[:, 0]), X[:, 0])
        return CustomSignal(
            signal=first + rng.normal(scale=0.01, size=X.shape[0]),
            direct_features=(feature_ids[0],),
            description="first feature plus seeded perturbation",
        )

    materialized = materialize_truth(CustomTruth(valid), population, seed=17)
    assert materialized.direct_features == ("a",)

    def invalid(
        X: np.ndarray, feature_ids: tuple[str, ...], rng: np.random.Generator
    ) -> CustomSignal:
        del feature_ids, rng
        return CustomSignal(signal=np.ones(X.shape[0] - 1), description="wrong length")

    with pytest.raises(ValueError, match="shape"):
        materialize_truth(CustomTruth(invalid), population, seed=17)
