"""Cross-interface scientific contracts and adversarial public inputs."""

import numpy as np
import pytest

import pyplasmode as ppm

pytestmark = pytest.mark.unit


def population() -> ppm.Population:
    return ppm.Population(np.random.default_rng(41).normal(size=(100, 4)), tuple("abcd"))


@pytest.mark.parametrize("weights", [(1.0, 0.0), (1.0, np.nan), (1.0, np.inf)])
def test_direct_truth_rejects_non_generating_weights(weights) -> None:
    with pytest.raises(ValueError, match="weights"):
        ppm.SparseTruth(features=("a", "b"), weights=weights)
    with pytest.raises(ValueError, match="weights"):
        ppm.CorrelatedTruth(("g1", "g2"), weights=weights)


def test_invalid_mechanism_names_fail_at_construction() -> None:
    with pytest.raises(ValueError, match="operation"):
        ppm.InteractionTruth(("a", "b"), operation="typo")
    with pytest.raises(ValueError, match="signal_form"):
        ppm.CorrelatedTruth(("g",), signal_form="typo")


def test_group_means_have_no_arbitrary_exact_targets() -> None:
    pop = population()
    groups = ppm.FeatureSets((ppm.FeatureSet("g", ("a", "b")),))
    first, second = (
        ppm.materialize_truth(
            ppm.CorrelatedTruth(("g",), signal_form="group_means"),
            pop,
            feature_sets=groups,
            seed=seed,
        )
        for seed in (1, 2)
    )
    np.testing.assert_array_equal(first.signal, second.signal)
    assert first.direct_features == second.direct_features == ()
    assert first.admissible_groups[0].direct_feature is None
    assert first.weights == (ppm.FeatureWeight("a", 0.5), ppm.FeatureWeight("b", 0.5))
    with pytest.raises(ValueError, match="exact"):
        ppm.evaluate_ranking(tuple("abcd"), first, depths=(1,))


def test_exact_recovery_is_representation_invariant_with_unequal_weights() -> None:
    truth = ppm.materialize_truth(
        ppm.SparseTruth(features=("a", "b"), weights=(9.0, 1.0)), population(), seed=1
    )
    nomination = ppm.nomination_from_scores(tuple("abcd"), [4, 3, 2, 1], depth=1)
    assert ppm.evaluate_ranking(tuple("abcd"), truth, depths=(1,)).at_depth[0].recall == 0.5
    assert ppm.evaluate_fractional_recovery(nomination, truth, estimand="exact").value == 0.5
    assert ppm.evaluate_fractional_recovery(nomination, truth, estimand="module").value == 0.9
    reference = ppm.evaluate_matched_recovery(
        nomination, truth, estimand="exact", strata=("one", "two", "two", "two")
    )
    assert reference.expected_recovery == 0.5


@pytest.mark.parametrize("size", [1, 100, 100_000])
def test_null_continuous_cannot_claim_positive_explained_variance(size) -> None:
    with pytest.raises(ppm.OutcomeSpecificationError, match="constant signal"):
        ppm.generate_outcome(
            ppm.ContinuousOutcome(0, 1, variance_explained=0.25),
            np.zeros(size),
            np.random.default_rng(1),
        )


def test_undefined_reconstruction_and_stability_fail() -> None:
    pop = population()
    for rows, signal in [([3], pop.X[:, 0]), ([3, 4], np.zeros(100))]:
        with pytest.raises(ValueError, match="R-squared"):
            ppm.evaluate_signal_reconstruction(
                pop.X,
                pop.feature_ids,
                selected_features=("a",),
                signal=signal,
                development_indices=[0, 1, 2],
                evaluation_indices=rows,
            )
    with pytest.raises(ValueError, match="two features"):
        ppm.rank_stability((("a",), ("a",)), depth=1)


def test_invalid_truth_metadata_fails_before_evaluation() -> None:
    with pytest.raises(ValueError, match="weights"):
        ppm.FeatureWeight("a", 0.0)
    for updates in [
        {"direct_features": ("unknown",)},
        {"direct_features": ("a", "a")},
        {"weights": (ppm.FeatureWeight("a", 1.0), ppm.FeatureWeight("a", 2.0))},
        {"kind": "misspelled"},
    ]:
        args = {"kind": "sparse", "signal": np.arange(5.0), "feature_universe": ("a", "b")}
        args.update(updates)
        with pytest.raises(ValueError):
            ppm.MaterializedTruth(**args)


@pytest.mark.parametrize("value", [True, 1.5])
def test_counts_and_depths_reject_non_integers(value) -> None:
    with pytest.raises(ValueError, match="integer"):
        ppm.PartitionSampleSizes(value, 2, 2)
    with pytest.raises(ValueError, match="integer"):
        ppm.nomination_from_scores(("a", "b"), [1, 0], depth=value)


def test_row_indices_are_not_silently_truncated() -> None:
    pop = population()
    with pytest.raises(ValueError, match="integer"):
        ppm.evaluate_signal_reconstruction(
            pop.X,
            pop.feature_ids,
            selected_features=("a",),
            signal=pop.X[:, 0],
            development_indices=[0.5, 1.5, 2.5],
            evaluation_indices=[3, 4],
        )


def test_constant_explicit_feature_is_not_a_generating_target() -> None:
    X = np.column_stack((np.arange(10.0), np.ones(10)))
    with pytest.raises(ValueError, match="constant"):
        ppm.materialize_truth(
            ppm.SparseTruth(features=("a", "b")), ppm.Population(X, ("a", "b")), seed=1
        )


def test_custom_input_mutation_is_rejected_without_changing_population() -> None:
    pop = population()
    original = pop.X.copy()

    def mutate(X, feature_ids, rng):
        X[0, 0] = 100
        return ppm.CustomSignal(X[:, 0], direct_features=("a",))

    with pytest.raises(ValueError, match="read-only"):
        ppm.materialize_truth(ppm.CustomTruth(mutate), pop, seed=1)
    np.testing.assert_array_equal(pop.X, original)


def test_rank_biased_overlap_matches_prefix_formula() -> None:
    rng = np.random.default_rng(9)
    left = tuple("abcdefgh")
    for persistence in (0.5, 0.9, 0.99):
        for _ in range(10):
            right = tuple(rng.permutation(left))
            agreements = [len(set(left[:d]) & set(right[:d])) / d for d in range(1, 9)]
            expected = (1 - persistence) * sum(
                agreement * persistence**index for index, agreement in enumerate(agreements)
            ) + agreements[-1] * persistence**8
            actual = ppm.rank_stability((left, right), depth=3, persistence=persistence)
            assert actual.mean_rank_biased_overlap == pytest.approx(expected)


def test_infeasible_logit_request_has_a_calibration_error() -> None:
    with pytest.raises(ppm.CalibrationError, match="bracket"):
        ppm.generate_outcome(
            ppm.BinaryOutcome(probability=1e-30, odds_ratio=1),
            np.zeros(10),
            np.random.default_rng(1),
        )


def test_group_specificity_matches_group_count_not_total_membership() -> None:
    pop = population()
    nomination = ppm.nomination_from_scores(pop.feature_ids, [4, 3, 2, 1], depth=1)
    own = ppm.MaterializedTruth(
        kind="correlated",
        signal=pop.X[:, 0],
        feature_universe=pop.feature_ids,
        admissible_groups=(ppm.RecoveryGroup("one", "a", ("a",)),),
    )
    foreign = ppm.MaterializedTruth(
        kind="correlated",
        signal=pop.X[:, 1],
        feature_universe=pop.feature_ids,
        admissible_groups=(ppm.RecoveryGroup("two", "b", ("b", "c")),),
    )
    result = ppm.evaluate_outcome_specificity(
        nomination, own, (foreign,), estimand="group", strata=("all",) * 4
    )
    assert result.specificity == pytest.approx(1.25)
