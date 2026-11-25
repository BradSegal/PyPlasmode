"""Selection agreement, censoring-aware discrimination and participant uncertainty.

Rankings and predictions must already share feature or participant identities.
These metrics do not fit models or move data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

from pyplasmode._validation import identities, integer
from pyplasmode.evaluate import compare_paired_replicates, rank_stability
from pyplasmode.nominations import FractionalNomination, nomination_from_scores

__all__ = [
    "BootstrapDifference",
    "ConsensusRanking",
    "DiscriminationGap",
    "SelectionStability",
    "consensus_ranking",
    "cumulative_dynamic_auc",
    "discrimination_gap",
    "learnability_gap",
    "paired_bootstrap_difference",
    "rank_biased_overlap",
    "selection_stability",
]


@dataclass(frozen=True, slots=True)
class SelectionStability:
    """Agreement of feature selections across refits after accounting for chance.

    Attributes:
        value: One for perfect agreement, zero at chance and negative below chance.
        feature_ids: Feature order from the first refit.
        frequencies: Mean selection probability per feature in ID order.
        membership_budgets: Total slots per refit in input order. Boundary ties share slots equally.

    The score is `None` when every refit selects nothing or every refit selects all features.
    """

    value: float | None
    feature_ids: tuple[str, ...]
    frequencies: tuple[float, ...]
    membership_budgets: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ConsensusRanking:
    """Combined ordering by selection frequency and rank when selected.

    Attributes:
        feature_ids: Input feature order for the ranks, probabilities and medians.
        ranking: Sorted feature IDs, alphabetised within ties.
        ranks: One-based comparison ranks, averaged within ties.
        inclusion_probabilities: Mean selection probability across refits.
        median_selected_ranks: Median rank when selectable, or infinity if never selectable.
    """

    feature_ids: tuple[str, ...]
    ranking: tuple[str, ...]
    ranks: tuple[float, ...]
    inclusion_probabilities: tuple[float, ...]
    median_selected_ranks: tuple[float, ...]


def selection_stability(nominations: Sequence[FractionalNomination]) -> SelectionStability:
    """Measure how consistently refits select the same features above chance.

    The Nogueira score subtracts from one the ratio of observed to chance selection variance.
    It averages over independent uniform resolutions of ties at each cutoff. It is negative
    when observed variance exceeds chance. Unequal numbers of slots cap the score below one.

    Args:
        nominations: At least two selections on the same set of at least two features, in any order.
    """
    if len(nominations) < 2 or len(nominations[0].feature_ids) < 2:
        raise ValueError("stability requires at least two refits and two features")
    features = nominations[0].feature_ids
    rows = []
    for item in nominations:
        if set(item.feature_ids) != set(features):
            raise ValueError("nominations must share one feature universe")
        weights = dict(zip(item.feature_ids, item.membership, strict=True))
        rows.append([weights[name] for name in features])
    probabilities = np.asarray(rows, dtype=np.float64)
    frequency = probabilities.mean(axis=0)
    budgets = probabilities.sum(axis=1)
    fraction = float(frequency.mean())
    value = None
    if 0.0 < fraction < 1.0:
        # Expected unbiased sample variance = variance of row means plus the
        # average variance introduced by independent tie resolutions.
        variance = probabilities.var(axis=0, ddof=1) + np.mean(
            probabilities * (1.0 - probabilities), axis=0
        )
        value = float(1.0 - variance.mean() / (fraction * (1.0 - fraction)))
    return SelectionStability(value, features, tuple(frequency.tolist()), tuple(budgets.tolist()))


def _ranks(values: ArrayLike) -> NDArray[np.float64]:
    ranks = np.asarray(values, dtype=np.float64)
    if ranks.ndim != 1 or ranks.size < 2 or not np.isfinite(ranks).all():
        raise ValueError("ranks must be a finite vector with at least two features")
    return ranks


def rank_biased_overlap(
    left_ranks: ArrayLike,
    right_ranks: ArrayLike,
    *,
    persistence: float = 0.9,
    tie_method: Literal["webber", "independent"] = "webber",
) -> float:
    """Measure agreement between two rankings, giving the top positions more weight.

    Args:
        left_ranks: Finite ranks for two or more features. Lower is better. Equal ranks are ties.
        right_ranks: Finite ranks for the same features in the same order.
        persistence: Value in (0, 1). Smaller values weight the top more.
        tie_method: `"webber"` includes whole ties. `"independent"` breaks ties at random.

    Returns:
        Weighted agreement in [0, 1]. For `"webber"`, each cutoff uses twice the shared count
        divided by the combined sizes of the two top lists. For `"independent"`, it averages the
        shared fraction over independent uniform resolutions, so identical tied lists can score
        below one. Position d has weight `(1 - p) * p**(d - 1)`, with the rest on full agreement.
    """
    left, right = _ranks(left_ranks), _ranks(right_ranks)
    if left.shape != right.shape:
        raise ValueError("rank vectors must have the same shape and feature order")
    if not np.isfinite(persistence) or not 0.0 < persistence < 1.0:
        raise ValueError("persistence must be finite and strictly between zero and one")
    if tie_method not in {"webber", "independent"}:
        raise ValueError("tie_method must be 'webber' or 'independent'")
    size = left.size
    if np.unique(left).size == size and np.unique(right).size == size:
        features = np.asarray([str(i) for i in range(size)])
        return rank_stability(
            (tuple(features[np.argsort(left)]), tuple(features[np.argsort(right)])),
            depth=1,
            persistence=persistence,
        ).mean_rank_biased_overlap
    first_left = rankdata(left, method="min")
    width_left = rankdata(left, method="max") - first_left + 1
    first_right = rankdata(right, method="min")
    width_right = rankdata(right, method="max") - first_right + 1
    total = 0.0
    for depth in range(1, size):
        if tie_method == "webber":
            a, b = first_left <= depth, first_right <= depth
            agreement = float(2.0 * np.count_nonzero(a & b) / (a.sum() + b.sum()))
        else:
            a = np.clip((depth - first_left + 1) / width_left, 0.0, 1.0)
            b = np.clip((depth - first_right + 1) / width_right, 0.0, 1.0)
            agreement = float(a @ b) / depth
        total += persistence ** (depth - 1) * agreement
    return float((1.0 - persistence) * total + persistence ** (size - 1))


def consensus_ranking(
    feature_ids: tuple[str, ...], scores: ArrayLike, *, depth: int
) -> ConsensusRanking:
    """Rank features by how often refits select them, then by their median rank when selected.

    Positive ties at the cutoff share the remaining slots equally.

    Args:
        feature_ids: Unique column IDs for `scores`.
        scores: Finite, non-negative importance scores, with one row per refit (at least two).
        depth: Number of selection slots per refit, from one to the feature count.
    """
    identities(feature_ids, "feature_ids")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(feature_ids):
        raise ValueError("scores must be an aligned refit-by-feature matrix")
    memberships = np.vstack(
        [nomination_from_scores(feature_ids, row, depth=depth).membership for row in values]
    )
    ranks = np.vstack([rankdata(-row, method="average") for row in values])
    frequencies = memberships.mean(axis=0)
    medians = np.asarray(
        [
            np.median(ranks[memberships[:, j] > 0, j])
            if np.any(memberships[:, j] > 0)
            else float("inf")
            for j in range(len(feature_ids))
        ]
    )
    keys = list(zip(-frequencies, medians, strict=True))
    evidence_order = {key: i for i, key in enumerate(sorted(set(keys)))}
    comparison = rankdata([evidence_order[key] for key in keys], method="average")
    order = sorted(range(len(feature_ids)), key=lambda j: (*keys[j], feature_ids[j]))
    return ConsensusRanking(
        feature_ids,
        tuple(feature_ids[j] for j in order),
        tuple(comparison.tolist()),
        tuple(frequencies.tolist()),
        tuple(medians.tolist()),
    )


def _survival(time: ArrayLike, event: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    times = np.asarray(time, dtype=np.float64)
    events = np.asarray(event)
    if (
        times.ndim != 1
        or times.size < 2
        or events.shape != times.shape
        or not np.isfinite(times).all()
        or np.any(times <= 0)
        or not np.isin(events, [0, 1]).all()
    ):
        raise ValueError("survival times and event indicators are invalid")
    return times, events.astype(np.bool_)


def _censoring_survival(
    time: NDArray[np.float64], event: NDArray[np.bool_], query: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Reverse Kaplan-Meier: observed events leave the risk set before censoring
    # at the same time. Evaluate the resulting right-continuous step function.
    unique, inverse, counts = np.unique(time, return_inverse=True, return_counts=True)
    events = np.bincount(inverse, weights=event, minlength=unique.size)
    at_risk = time.size - np.r_[0, np.cumsum(counts[:-1])]
    denominator = at_risk - events
    decrement = np.divide(
        counts - events, denominator, out=np.zeros(unique.size), where=denominator > 0
    )
    survival = np.r_[1.0, np.cumprod(1.0 - decrement)]
    return np.asarray(survival[np.searchsorted(unique, query, side="right")], dtype=np.float64)


def cumulative_dynamic_auc(
    training_time: ArrayLike,
    training_event: ArrayLike,
    time: ArrayLike,
    event: ArrayLike,
    risk: ArrayLike,
    *,
    horizon: float,
    tied_tolerance: float = 0.0,
) -> float:
    """Measure how well risk scores separate events by a cutoff from known event-free observations.

    Cases have an observed event by the cutoff. Controls have follow-up beyond it. Censoring
    by the cutoff is excluded. Under independent censoring, each case is weighted by the inverse
    probability of remaining uncensored at its event time, estimated by reverse Kaplan-Meier
    on the training rows. Administrative follow-up must extend beyond the evaluation horizon.

    Args:
        training_time: Positive finite follow-up for at least two training rows.
        training_event: Training indicators in row order, zero for censoring or one for event.
        time: Positive finite follow-up for at least two test rows.
        event: Test indicators in row order, zero for censoring or one for event.
        risk: Finite scores increasing with event risk, in test row order.
        horizon: Positive cutoff in the same units as both time vectors.
        tied_tolerance: Finite non-negative gap for chaining adjacent scores into ties.

    Returns:
        Weighted share of case-control pairs with higher case risk, giving ties half credit.

    Raises:
        ValueError: No cases or controls, the horizon reaches maximum training follow-up, or
            estimated censoring survival is zero at a case time or the horizon.
    """
    train_t, train_e = _survival(training_time, training_event)
    times, events = _survival(time, event)
    scores = np.asarray(risk, dtype=np.float64)
    if scores.shape != times.shape or not np.isfinite(scores).all():
        raise ValueError("risk must be a finite vector aligned with evaluation time")
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    if horizon >= train_t.max():
        raise ValueError("horizon must be strictly earlier than maximum training follow-up time")
    if not np.isfinite(tied_tolerance) or tied_tolerance < 0:
        raise ValueError("tied_tolerance must be finite and non-negative")
    cases = events & (times <= horizon)
    controls = times > horizon
    if not cases.any() or not controls.any():
        raise ValueError("AUROC requires cases and controls at the horizon")
    censor = _censoring_survival(train_t, train_e, np.r_[times[cases], horizon])
    if np.any(censor <= 0):
        raise ValueError("censoring survival must be positive at cases and horizon")
    weights = np.zeros(times.size)
    weights[cases] = 1.0 / censor[:-1]
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order]) > tied_tolerance) + 1]
    case_mass = np.add.reduceat(weights[order], starts)
    control_mass = np.add.reduceat(controls[order].astype(float), starts)
    lower_controls = np.cumsum(control_mass) - control_mass
    return float(case_mass @ (lower_controls + 0.5 * control_mass) / weights.sum() / controls.sum())


@dataclass(frozen=True, slots=True)
class BootstrapDifference:
    """Paired bootstrap difference and percentile interval.

    Attributes:
        left: Left estimate on the original sample.
        right: Right estimate on the original sample.
        difference: Right minus left on the original sample.
        confidence_level: Requested coverage in (0, 1).
        confidence_interval_low: Lower percentile bound across successful draws.
        confidence_interval_high: Upper percentile bound across successful draws.
        differences: Right minus left for each draw in order, with `None` for failures.
        failed_resamples: Draw count with either estimate undefined.
    """

    left: float
    right: float
    difference: float
    confidence_level: float
    confidence_interval_low: float
    confidence_interval_high: float
    differences: tuple[float | None, ...]
    failed_resamples: int


def paired_bootstrap_difference(
    participants: int | Sequence[str] | NDArray[np.str_],
    statistic: Callable[[NDArray[np.int64]], tuple[float, float] | None],
    *,
    resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapDifference:
    """Estimate uncertainty for a metric difference using shared participant resamples.

    Each draw samples the original row count with replacement. Keep fitted models fixed
    and re-estimate evaluation nuisance parameters in the callback when needed. Training
    samples remain fixed unless the callback resamples them.

    Args:
        participants: Row count of at least two, or unique participant IDs in row order.
        statistic: Takes zero-based row indices and returns finite left and right estimates,
            or `None` if either is undefined. Use the same indices for both estimates.
        resamples: Number of attempted draws, at least two.
        confidence_level: Percentile interval coverage in (0, 1).
        seed: Non-negative sampling seed.

    Raises:
        ValueError: The original comparison is undefined or fewer than two draws succeed.
    """
    if isinstance(participants, (int, np.integer)):
        integer(participants, "participants", minimum=2)
        size = int(participants)
    else:
        if isinstance(participants, np.ndarray):
            if participants.ndim != 1:
                raise ValueError("participant IDs must be one-dimensional")
            participants = participants.tolist()
        if not isinstance(participants, Sequence) or isinstance(participants, (str, bytes)):
            raise ValueError("participants must be an integer or a sequence of unique IDs")
        participant_ids = tuple(participants)
        identities(participant_ids, "participants")
        size = len(participant_ids)
    integer(resamples, "resamples", minimum=2)
    integer(seed, "seed", minimum=0)
    if size < 2 or not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap requires participants and a valid confidence level")

    def evaluate(indices: NDArray[np.int64]) -> tuple[float, float] | None:
        value = statistic(indices)
        if value is not None and (len(value) != 2 or not np.isfinite(value).all()):
            raise ValueError("statistic must return two finite estimates or None")
        return value

    point = evaluate(np.arange(size, dtype=np.int64))
    if point is None:
        raise ValueError("the original paired statistic is undefined")
    rng = np.random.default_rng(seed)
    left: list[float | None] = []
    right: list[float | None] = []
    for _ in range(resamples):
        pair = evaluate(rng.integers(size, size=size))
        left.append(None if pair is None else float(pair[0]))
        right.append(None if pair is None else float(pair[1]))
    ids = tuple(str(i) for i in range(resamples))
    # Align failed pairs and take percentile limits from the paired bootstrap draws.
    comparison = compare_paired_replicates(ids, left, ids, right)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(comparison.differences, [alpha, 1.0 - alpha])
    differences = tuple(
        None if a is None or b is None else b - a for a, b in zip(left, right, strict=True)
    )
    return BootstrapDifference(
        float(point[0]),
        float(point[1]),
        float(point[1] - point[0]),
        confidence_level,
        float(low),
        float(high),
        differences,
        comparison.failed_pair_count,
    )


@dataclass(frozen=True, slots=True)
class DiscriminationGap:
    """Difference in AUC between the generating model and a fitted model on the same outcomes.

    Attributes:
        oracle_auc: AUC of the known generating model's risk scores.
        fitted_auc: AUC of the fitted model's risk scores.
        gap: Generating-model AUC minus fitted-model AUC. Negative values are possible.
    """

    oracle_auc: float
    fitted_auc: float
    gap: float


def discrimination_gap(
    oracle_risk: ArrayLike, fitted_risk: ArrayLike, *, metric: Callable[[ArrayLike], float]
) -> DiscriminationGap:
    """Compare risk scores from the known generating model and a fitted model using one shared AUC.

    Args:
        oracle_risk: At least two finite scores from the generating model, in evaluation row order.
        fitted_risk: Finite fitted-model scores in the same row order.
        metric: Function mapping a score vector to AUC in [0, 1], with the outcomes fixed.
            For survival, also fix the horizon and censoring reference.

    A finite sample may give a negative gap.
    """
    oracle = np.asarray(oracle_risk, dtype=np.float64)
    fitted = np.asarray(fitted_risk, dtype=np.float64)
    if oracle.ndim != 1 or oracle.size < 2 or oracle.shape != fitted.shape:
        raise ValueError("oracle and fitted risks must be aligned vectors")
    if not np.isfinite(oracle).all() or not np.isfinite(fitted).all():
        raise ValueError("risk scores must be finite")
    a, b = float(metric(oracle)), float(metric(fitted))
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("metric must return finite AUROCs in [0, 1]")
    return DiscriminationGap(a, b, a - b)


# Alternative name for the same oracle-minus-fitted difference.
learnability_gap = discrimination_gap
