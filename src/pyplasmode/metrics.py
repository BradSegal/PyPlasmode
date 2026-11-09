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
    """Expected Nogueira stability, including independent boundary-tie resolution.

    ``value`` is None when every refit selects no features or all features.
    ``membership_budgets`` count slots; tied features can occupy fractional slots.
    """

    value: float | None
    feature_ids: tuple[str, ...]
    frequencies: tuple[float, ...]
    membership_budgets: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ConsensusRanking:
    """Inclusion-first consensus; equal evidence retains equal comparison ranks.

    Fields are aligned to ``feature_ids``. ``ranking`` uses feature names only
    to order storage within an evidence tie; use ``ranks`` for comparisons.
    """

    feature_ids: tuple[str, ...]
    ranking: tuple[str, ...]
    ranks: tuple[float, ...]
    inclusion_probabilities: tuple[float, ...]
    median_selected_ranks: tuple[float, ...]


def selection_stability(nominations: Sequence[FractionalNomination]) -> SelectionStability:
    """Return chance-corrected top-k stability allowing unequal achieved budgets.

    Uses Definition 4 of Nogueira, Sechidis and Brown (JMLR 2018). For fractional
    memberships, averages the binary-selection variance over independent uniform
    resolutions of each refit's boundary tie. Treating membership weights as
    observed binary selections would incorrectly reward uninformative ties.
    Unequal row budgets cap maximum stability below one. Negative values are
    possible and are not clipped. Requires at least two
    refits with the same universe of at least two features; orders may differ.
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
    """Return finite extrapolated RBO on aligned complete rank vectors.

    Lower values mean better ranks; equal values indicate ties. Convert higher
    importance scores with ``rankdata(-scores, method="average")`` first.
    The default ``tie_method="webber"`` follows Webber et al. (2010), section
    4.3 (4.6, Eq. 28 in the author's version): prefixes contain whole tie groups
    starting at or before depth d, and
    agreement is twice their intersection divided by their combined sizes.
    Identical tied rankings have agreement one.

    ``tie_method="independent"`` instead averages over independent uniform
    tie-breaking permutations. An all-tied ranking then has self-agreement below
    one. This is a different estimand from the published tie convention.
    Prefix weights are (1-p)*p**(d-1), with complete-list tail p**(n-1).
    Requires equal complete universes and 0 < persistence < 1.
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
    """Order features by mean top-k membership, then median selected rank.

    ``scores`` is a refit-by-feature matrix of non-negative importances. Positive
    boundary ties use ``nomination_from_scores``. Median ranks use only refits
    with positive membership; never-selected features receive infinity. Evidence
    ties remain tied. At least two refits are required; depth may span the universe.
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
    """Estimate cumulative/dynamic AUROC at one horizon using training IPCW.

    Cases have observed events at or before horizon; controls have observed time
    strictly beyond it. Earlier censoring is excluded. Cases receive 1/G(time),
    using a reverse Kaplan-Meier censoring curve from the training sample.
    Controls' common 1/G(horizon) cancels. Independent censoring is assumed.
    All time arrays and horizon must use the same units. Higher risk means a
    higher event risk. The default uses exact score ties, credited one half.
    An explicit positive tied_tolerance uses adjacent-score tie blocks, including
    transitive chains, for compatibility with scikit-survival. Use it only when
    that historical convention is intended; it can merge dense probabilities.

    Raises ValueError for missing cases/controls, horizon outside training
    follow-up, zero censoring support or invalid arrays. Events exactly at the
    horizon are cases; censoring exactly there is not a control. Administrative
    follow-up must extend beyond the evaluation horizon.
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
    """Right-minus-left participant bootstrap with a percentile interval.

    Undefined draws are retained as None and excluded jointly. The interval is
    conditional on successful draws; attempted/failed counts remain explicit.
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
    """Resample participants jointly and return right-minus-left uncertainty.

    The callback receives the same integer row indices for both methods and
    returns their two estimates, or None when either estimate is undefined.
    It must preserve method pairing and fixed fitted models; it may re-estimate
    evaluation nuisance parameters if appropriate. Training samples remain fixed
    unless the callback explicitly resamples them. Supply a row count or a sequence
    of unique IDs, including a one-dimensional NumPy string array. Exactly
    resamples draws are attempted; at least two must succeed. Exceptions and
    non-finite estimates are errors, not silently discarded failures.

    This percentile interval describes participant sampling for fixed models.
    Use compare_paired_replicates for independent simulation-world comparisons.
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
    # Reuse identity/failure alignment, but deliberately do not use its Monte
    # Carlo t interval: bootstrap draws are not independent simulation worlds.
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
    """Oracle-minus-fitted AUROC on the same evaluation population."""

    oracle_auc: float
    fitted_auc: float
    gap: float


def discrimination_gap(
    oracle_risk: ArrayLike, fitted_risk: ArrayLike, *, metric: Callable[[ArrayLike], float]
) -> DiscriminationGap:
    """Compare oracle and fitted scores using one fixed AUROC evaluator.

    Bind shared labels, horizon, censoring reference and sample in metric, e.g.
    a closure over cumulative_dynamic_auc or sklearn.metrics.roc_auc_score.
    The gap is oracle minus fitted; a finite sample may give a negative gap.
    No refitting or optimisation is performed.
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
