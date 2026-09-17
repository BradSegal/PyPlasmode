"""Feature selections with tied scores and recovery against matched random selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import comb, factorial
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import hypergeom

from pyplasmode._validation import identities, integer
from pyplasmode.results import FractionalRecovery, MatchedRecovery, OutcomeSpecificity
from pyplasmode.truth import MaterializedTruth

type RecoveryEstimand = Literal["exact", "group", "module", "interaction"]


@dataclass(frozen=True, slots=True)
class FractionalNomination:
    """A feature selection with fractional membership for tied positive scores."""

    feature_ids: tuple[str, ...]
    membership: NDArray[np.float64]
    requested_depth: int
    positive_support: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require aligned fractional membership without hidden padding."""
        integer(self.requested_depth, "requested_depth", minimum=1)
        identities(self.feature_ids, "feature_ids")
        identities(self.positive_support, "positive_support", allow_empty=True)
        values = np.asarray(self.membership, dtype=np.float64)
        support = set(self.positive_support)
        positions = {feature: index for index, feature in enumerate(self.feature_ids)}
        boundary = values[(values > 0.0) & (values < 1.0)]
        if (
            values.shape != (len(self.feature_ids),)
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
            or self.requested_depth > len(self.feature_ids)
            or not support.issubset(positions)
            or any(
                values[index] > 0.0 and feature not in support
                for index, feature in enumerate(self.feature_ids)
            )
            or (
                boundary.size > 0
                and (
                    not np.allclose(boundary, boundary[0], atol=1e-12, rtol=0.0)
                    or not np.isclose(float(np.sum(boundary)), round(float(np.sum(boundary))))
                )
            )
            or not np.isclose(
                float(np.sum(values)),
                min(self.requested_depth, len(self.positive_support)),
                atol=1e-12,
            )
        ):
            raise ValueError("fractional nomination is invalid")
        frozen = values.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "membership", frozen)

    @property
    def achieved_size(self) -> int:
        """Return the number of features with non-zero selection membership."""
        return int(np.count_nonzero(self.membership > 0.0))

    @property
    def selected_features(self) -> tuple[str, ...]:
        """Return the depth-limited features with non-zero selection probability."""
        return tuple(
            feature
            for feature, weight in zip(self.feature_ids, self.membership, strict=True)
            if weight > 0.0
        )

    @property
    def membership_budget(self) -> float:
        """Return the sum of feature-selection membership probabilities."""
        return float(np.sum(self.membership))


def nomination_from_scores(
    feature_ids: tuple[str, ...],
    scores: ArrayLike,
    *,
    depth: int,
) -> FractionalNomination:
    """Select leading positive scores, sharing remaining slots among tied features.

    Args:
        feature_ids: Unique feature identities aligned to scores.
        scores: Finite non-negative method-specific importance scores.
        depth: Requested number of leading features.

    Returns:
        Feature membership probabilities summing to the smaller of depth and
        the number of positive scores. Zero-score features are not selected.

    Raises:
        ValueError: If identities, scores, or depth are invalid.
    """
    values = np.asarray(scores, dtype=np.float64)
    integer(depth, "depth", minimum=1)
    identities(feature_ids, "feature_ids")
    if (
        values.shape != (len(feature_ids),)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or depth > len(feature_ids)
    ):
        raise ValueError("scores and feature identities are invalid")
    positive = values > 0.0
    support = tuple(feature for feature, keep in zip(feature_ids, positive, strict=True) if keep)
    membership = np.zeros(values.shape, dtype=np.float64)
    if len(support) <= depth:
        membership[positive] = 1.0
        return FractionalNomination(feature_ids, membership, depth, support)
    ordered = np.sort(values[positive])[::-1]
    boundary = float(ordered[depth - 1])
    above = values > boundary
    tied = values == boundary
    membership[above] = 1.0
    remaining = depth - int(np.count_nonzero(above))
    membership[tied] = remaining / int(np.count_nonzero(tied))
    return FractionalNomination(feature_ids, membership, depth, support)


def _membership(nomination: FractionalNomination) -> dict[str, float]:
    return dict(zip(nomination.feature_ids, nomination.membership.tolist(), strict=True))


def _boundary_selection_probability(
    membership_values: tuple[float, ...],
    target_values: tuple[float, ...],
    *,
    require_all: bool,
) -> float:
    """Integrate nonlinear recovery over one uniform boundary tie."""
    if require_all and any(value == 0.0 for value in target_values):
        return 0.0
    if not require_all and any(value == 1.0 for value in target_values):
        return 1.0
    fractional_values = tuple(value for value in membership_values if 0.0 < value < 1.0)
    target_fractional = sum(0.0 < value < 1.0 for value in target_values)
    if target_fractional == 0:
        return 1.0 if require_all else 0.0
    boundary_size = len(fractional_values)
    boundary_slots = round(sum(fractional_values))
    if require_all:
        if boundary_slots < target_fractional:
            return 0.0
        return float(
            comb(boundary_size - target_fractional, boundary_slots - target_fractional)
            / comb(boundary_size, boundary_slots)
        )
    if boundary_size - target_fractional < boundary_slots:
        return 1.0
    return float(
        1.0
        - comb(boundary_size - target_fractional, boundary_slots)
        / comb(boundary_size, boundary_slots)
    )


def evaluate_fractional_recovery(
    nomination: FractionalNomination,
    truth: MaterializedTruth,
    *,
    estimand: RecoveryEstimand,
) -> FractionalRecovery:
    """Score a feature selection against the chosen recovery target.

    Args:
        nomination: Feature membership probabilities, including tied scores.
        truth: Materialized generating mechanism on the same feature universe.
        estimand: Exact, group, module, or interaction recovery definition.

    Returns:
        Bounded recovery with explicit truth and nomination support.

    Raises:
        ValueError: If the nomination and truth differ or the estimand has no target.
    """
    if nomination.feature_ids != truth.feature_universe:
        raise ValueError("nomination and truth feature universes differ")
    membership = _membership(nomination)
    if estimand == "exact":
        if not truth.direct_features:
            raise ValueError("exact recovery requires direct truth features")
        value = float(np.mean([membership[feature] for feature in truth.direct_features]))
        truth_support = len(truth.direct_features)
    elif estimand == "group":
        if not truth.admissible_groups:
            raise ValueError("group recovery requires admissible truth groups")
        recovered = [
            _boundary_selection_probability(
                tuple(membership.values()),
                tuple(membership[feature] for feature in group.features),
                require_all=False,
            )
            for group in truth.admissible_groups
        ]
        value = float(np.mean(recovered))
        truth_support = len(truth.admissible_groups)
    elif estimand == "module":
        weights = {item.feature: abs(item.weight) for item in truth.weights}
        if not weights or sum(weights.values()) <= 0.0:
            raise ValueError("module recovery requires non-zero truth weights")
        value = sum(membership[feature] * weight for feature, weight in weights.items()) / sum(
            weights.values()
        )
        truth_support = len(weights)
    elif estimand == "interaction":
        if len(truth.direct_features) != 2:
            raise ValueError("interaction recovery requires two direct truth features")
        value = _boundary_selection_probability(
            tuple(membership.values()),
            tuple(membership[feature] for feature in truth.direct_features),
            require_all=True,
        )
        truth_support = 2
    else:
        raise ValueError(f"unsupported recovery estimand: {estimand}")
    return FractionalRecovery(
        estimand,
        float(value),
        truth_support,
        len(nomination.positive_support),
        nomination.requested_depth,
        nomination.achieved_size,
    )


def _state_count(weights: list[float], available: int, positive_support: int) -> int:
    counts = Counter(weights)
    selected = len(weights)
    if selected > positive_support or positive_support > available:
        raise ValueError("matched stratum support geometry is infeasible")
    denominator = factorial(available - positive_support) * factorial(positive_support - selected)
    for count in counts.values():
        denominator *= factorial(count)
    return factorial(available) // denominator


def _geometry(
    nomination: FractionalNomination, strata: tuple[str, ...]
) -> tuple[
    dict[str, tuple[int, ...]],
    dict[str, int],
    dict[str, tuple[float, ...]],
]:
    if len(strata) != len(nomination.feature_ids) or any(not item for item in strata):
        raise ValueError("strata must identify every feature")
    available: dict[str, list[int]] = defaultdict(list)
    support_counts: Counter[str] = Counter()
    weights: dict[str, list[float]] = defaultdict(list)
    positive = set(nomination.positive_support)
    for index, (stratum, weight) in enumerate(zip(strata, nomination.membership, strict=True)):
        available[stratum].append(index)
        if nomination.feature_ids[index] in positive:
            support_counts[stratum] += 1
        if weight > 0.0:
            weights[stratum].append(float(weight))
    return (
        {key: tuple(value) for key, value in available.items()},
        {key: support_counts[key] for key in available},
        {key: tuple(value) for key, value in weights.items()},
    )


def _sample_nomination(
    nomination: FractionalNomination,
    available: dict[str, tuple[int, ...]],
    support_counts: dict[str, int],
    weights: dict[str, tuple[float, ...]],
    rng: np.random.Generator,
) -> FractionalNomination:
    membership = np.zeros(len(nomination.feature_ids), dtype=np.float64)
    positive_indices: list[int] = []
    for key in sorted(available):
        values = np.asarray(weights.get(key, ()), dtype=np.float64)
        support_count = support_counts[key]
        if support_count == 0:
            continue
        support = np.asarray(
            rng.choice(available[key], size=support_count, replace=False),
            dtype=np.int64,
        )
        positive_indices.extend(int(index) for index in support)
        if values.size:
            chosen = rng.choice(support, size=values.size, replace=False)
            membership[chosen] = rng.permutation(values)
    positive_set = set(positive_indices)
    return FractionalNomination(
        nomination.feature_ids,
        membership,
        nomination.requested_depth,
        tuple(
            feature for index, feature in enumerate(nomination.feature_ids) if index in positive_set
        ),
    )


def _expected_membership(
    index: int,
    *,
    available: dict[str, tuple[int, ...]],
    weights: dict[str, tuple[float, ...]],
) -> float:
    """Return marginal selection probability under the matched assignment law."""
    for key, indices in available.items():
        if index in indices:
            return float(sum(weights.get(key, ())) / len(indices))
    raise RuntimeError("feature index is absent from matched-reference strata")


def _no_target_selection_probability(
    target_indices: tuple[int, ...],
    *,
    available: dict[str, tuple[int, ...]],
    weights: dict[str, tuple[float, ...]],
) -> float:
    """Return the analytic probability that no target enters the final tied panel."""
    targets = set(target_indices)
    boundary_weights = tuple(
        weight for values in weights.values() for weight in values if 0.0 < weight < 1.0
    )
    boundary_count = len(boundary_weights)
    boundary_slots = round(sum(boundary_weights))
    target_boundary_distribution = np.asarray([1.0], dtype=np.float64)
    for key in sorted(available):
        indices = available[key]
        population = len(indices)
        target_count = sum(index in targets for index in indices)
        values = weights.get(key, ())
        full_count = sum(weight == 1.0 for weight in values)
        stratum_boundary_count = sum(0.0 < weight < 1.0 for weight in values)
        no_full_probability = float(hypergeom.pmf(0, population, target_count, full_count))
        maximum_target_boundary = min(
            target_count,
            stratum_boundary_count,
            population - full_count,
        )
        boundary_targets = np.arange(maximum_target_boundary + 1, dtype=np.int64)
        distribution = (
            np.zeros(boundary_targets.size, dtype=np.float64)
            if no_full_probability == 0.0
            else no_full_probability
            * np.asarray(
                hypergeom.pmf(
                    boundary_targets,
                    population - full_count,
                    target_count,
                    stratum_boundary_count,
                ),
                dtype=np.float64,
            )
        )
        target_boundary_distribution = np.convolve(
            target_boundary_distribution,
            distribution,
        )
    if boundary_slots == 0:
        probability = float(target_boundary_distribution[0])
    else:
        target_counts = np.arange(target_boundary_distribution.size, dtype=np.int64)
        avoid_boundary_targets = np.asarray(
            hypergeom.pmf(0, boundary_count, target_counts, boundary_slots),
            dtype=np.float64,
        )
        probability = float(target_boundary_distribution @ avoid_boundary_targets)
    if not np.isfinite(probability) or probability < -1e-12 or probability > 1.0 + 1e-12:
        raise RuntimeError("analytic matched-reference probability is invalid")
    return float(np.clip(probability, 0.0, 1.0))


def _expected_matched_recovery(
    truth: MaterializedTruth,
    *,
    estimand: RecoveryEstimand,
    available: dict[str, tuple[int, ...]],
    weights: dict[str, tuple[float, ...]],
    positions: dict[str, int],
) -> float:
    """Calculate the matched-reference expectation without simulation error."""
    if estimand == "exact":
        expected = np.asarray(
            [
                _expected_membership(
                    positions[feature],
                    available=available,
                    weights=weights,
                )
                for feature in truth.direct_features
            ],
            dtype=np.float64,
        )
        return float(np.mean(expected))
    if estimand == "module":
        target_weights = np.asarray(
            [abs(item.weight) for item in truth.weights],
            dtype=np.float64,
        )
        expected = np.asarray(
            [
                _expected_membership(
                    positions[item.feature],
                    available=available,
                    weights=weights,
                )
                for item in truth.weights
            ],
            dtype=np.float64,
        )
        return float(np.average(expected, weights=target_weights))
    if estimand == "group":
        recovered = tuple(
            1.0
            - _no_target_selection_probability(
                tuple(positions[feature] for feature in group.features),
                available=available,
                weights=weights,
            )
            for group in truth.admissible_groups
        )
        return float(np.mean(recovered))
    if estimand == "interaction":
        first, second = (positions[feature] for feature in truth.direct_features)
        neither_first = _no_target_selection_probability(
            (first,),
            available=available,
            weights=weights,
        )
        neither_second = _no_target_selection_probability(
            (second,),
            available=available,
            weights=weights,
        )
        neither = _no_target_selection_probability(
            (first, second),
            available=available,
            weights=weights,
        )
        return float(np.clip(1.0 - neither_first - neither_second + neither, 0.0, 1.0))
    raise ValueError(f"unsupported recovery estimand: {estimand}")


def sample_matched_nominations(
    nomination: FractionalNomination,
    *,
    strata: tuple[str, ...],
    draws: int,
    seed: int,
) -> tuple[FractionalNomination, ...]:
    """Randomise feature selections within strata while preserving counts and ties.

    Args:
        nomination: Observed feature-selection memberships and positive scores.
        strata: A group label per feature, defined without outcomes or generating truth.
        draws: Positive number of independent random selections.
        seed: Non-negative random seed.

    Returns:
        Selections with the same positive-score counts and membership weights
        in each stratum, reassigned to randomly chosen feature identities.

    Raises:
        ValueError: If geometry, draw count, or seed is invalid.
    """
    integer(draws, "draws", minimum=1)
    integer(seed, "seed")
    available, support_counts, weights = _geometry(nomination, strata)
    if any(
        len(weights.get(key, ())) > support_counts[key] or support_counts[key] > len(indices)
        for key, indices in available.items()
    ):
        raise ValueError("matched nomination geometry is not feasible")
    rng = np.random.default_rng(seed)
    return tuple(
        _sample_nomination(nomination, available, support_counts, weights, rng)
        for _ in range(draws)
    )


def evaluate_matched_recovery(
    nomination: FractionalNomination,
    truth: MaterializedTruth,
    *,
    estimand: RecoveryEstimand,
    strata: tuple[str, ...],
) -> MatchedRecovery:
    """Compare observed recovery with expected recovery under matched random selection.

    Args:
        nomination: Observed feature-selection memberships and positive scores.
        truth: Materialized truth used only by the recovery evaluator.
        estimand: Mechanism-specific recovery definition.
        strata: A group label per feature, defined without outcomes or generating truth.

    Returns:
        Observed recovery, its analytically calculated random expectation,
        and their difference.

    Raises:
        ValueError: If geometry or truth are invalid.
    """
    observed = evaluate_fractional_recovery(nomination, truth, estimand=estimand)
    available, support_counts, weights = _geometry(nomination, strata)
    positions = {feature: index for index, feature in enumerate(nomination.feature_ids)}
    panel_count = 1
    for key in sorted(available):
        stratum_weights = list(weights.get(key, ()))
        panel_count *= _state_count(
            stratum_weights,
            len(available[key]),
            support_counts[key],
        )
    expected = _expected_matched_recovery(
        truth,
        estimand=estimand,
        available=available,
        weights=weights,
        positions=positions,
    )
    return MatchedRecovery(
        estimand,
        "analytic",
        nomination.requested_depth,
        nomination.achieved_size,
        observed.value,
        expected,
        observed.value - expected,
        panel_count,
    )


def _recovery_target_features(
    truth: MaterializedTruth,
    estimand: RecoveryEstimand,
) -> frozenset[str]:
    """Return every feature that can receive credit under one recovery estimand."""
    if estimand in {"exact", "interaction"}:
        features = truth.direct_features
    elif estimand == "group":
        features = tuple(feature for group in truth.admissible_groups for feature in group.features)
    elif estimand == "module":
        features = tuple(item.feature for item in truth.weights)
    else:  # pragma: no cover - exhaustive over RecoveryEstimand
        raise ValueError(f"unsupported recovery estimand: {estimand}")
    if not features:
        raise ValueError("specificity requires a non-empty recovery target")
    return frozenset(features)


def evaluate_outcome_specificity(
    nomination: FractionalNomination,
    own_truth: MaterializedTruth,
    foreign_truths: tuple[MaterializedTruth, ...],
    *,
    estimand: RecoveryEstimand,
    strata: tuple[str, ...],
) -> OutcomeSpecificity:
    """Compare recovery of the model's generating signal with other disjoint signals.

    Args:
        nomination: Feature selection to evaluate unchanged against each signal.
        own_truth: Generating signal for the outcome used to fit the model.
        foreign_truths: Other signals of the same mechanism and recovery-target count.
        estimand: Mechanism-specific exact, group, module, or interaction recovery.
        strata: Outcome-blind matching stratum for every feature.

    Returns:
        Own adjusted recovery, mean foreign adjusted recovery, and their difference.

    Raises:
        ValueError: If truths differ in universe or mechanism, are absent, or overlap.
    """
    if not foreign_truths:
        raise ValueError("specificity requires at least one foreign truth")
    all_truths = (own_truth, *foreign_truths)
    if any(
        truth.kind != own_truth.kind or truth.feature_universe != own_truth.feature_universe
        for truth in foreign_truths
    ):
        raise ValueError("specificity truths differ in mechanism or feature universe")
    supports = tuple(_recovery_target_features(truth, estimand) for truth in all_truths)
    target_sizes = (
        {len(truth.admissible_groups) for truth in all_truths}
        if estimand == "group"
        else {len(support) for support in supports}
    )
    if len(target_sizes) != 1:
        raise ValueError("specificity truths must have the same target size")
    if any(left & right for index, left in enumerate(supports) for right in supports[index + 1 :]):
        raise ValueError("specificity truth targets are not disjoint")
    own = evaluate_matched_recovery(
        nomination,
        own_truth,
        estimand=estimand,
        strata=strata,
    ).chance_adjusted_recovery
    foreign = tuple(
        evaluate_matched_recovery(
            nomination,
            truth,
            estimand=estimand,
            strata=strata,
        ).chance_adjusted_recovery
        for truth in foreign_truths
    )
    mean_foreign = float(np.mean(foreign))
    return OutcomeSpecificity(
        estimand,
        own,
        mean_foreign,
        own - mean_foreign,
        len(foreign),
    )
