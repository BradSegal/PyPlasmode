"""Define the generating score, which outcomes are simulated from, and the recovery targets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pyplasmode._validation import identities, integer
from pyplasmode.data import Population
from pyplasmode.structures import FeatureSets

type TruthKind = Literal["null", "sparse", "correlated", "module", "interaction", "custom"]


@dataclass(frozen=True, slots=True)
class RecoveryGroup:
    """A group counted as found when at least one member is selected.

    Attributes:
        name: Group identifier.
        direct_feature: Designated generating feature, or `None` if there is none.
        features: Unique eligible feature IDs, including the direct feature if designated.
    """

    name: str
    direct_feature: str | None
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        identities((self.name,), "group name")
        identities(self.features, "group features")
        if self.direct_feature is not None and self.direct_feature not in self.features:
            raise ValueError("group direct feature must belong to its members")


@dataclass(frozen=True, slots=True)
class FeatureWeight:
    """A signed feature weight whose magnitude counts towards weighted recovery.

    Attributes:
        feature: Feature ID.
        weight: Finite, non-zero weight. In built-in additive generating scores it multiplies a
            standardised
            feature before the score itself is standardised.
    """

    feature: str
    weight: float

    def __post_init__(self) -> None:
        identities((self.feature,), "weight feature")
        if not np.isfinite(self.weight) or self.weight == 0.0:
            raise ValueError("truth weights must be finite and non-zero")


@dataclass(frozen=True, slots=True)
class MaterializedTruth:
    """The generating score, one number per row, and the targets for scoring feature selection.

    Attributes:
        kind: Signal structure, such as `"sparse"` or `"custom"`.
        signal: Read-only score in input row order.
        feature_universe: All input feature IDs in column order.
        direct_features: IDs counted as exact selection targets.
        admissible_groups: Groups credited when a selection includes any member.
        weights: Signed coefficients for scoring coverage by absolute weight.
        graph_region: Target graph node IDs.
        center: Mean before score standardisation.
        scale: Population standard deviation before score standardisation.
        description: Label for the signal structure.

    Unused target tuples are empty. For a score built by `materialize_truth`,
    `center` and `scale` are both zero in the null case.
    """

    kind: TruthKind
    signal: NDArray[np.float64]
    feature_universe: tuple[str, ...]
    direct_features: tuple[str, ...] = ()
    admissible_groups: tuple[RecoveryGroup, ...] = ()
    weights: tuple[FeatureWeight, ...] = ()
    graph_region: tuple[str, ...] = ()
    center: float = 0.0
    scale: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"null", "sparse", "correlated", "module", "interaction", "custom"}:
            raise ValueError("unsupported truth kind")
        identities(self.feature_universe, "feature universe")
        identities(self.direct_features, "direct features", allow_empty=True)
        identities(
            tuple(item.feature for item in self.weights), "weighted features", allow_empty=True
        )
        identities(
            tuple(group.name for group in self.admissible_groups), "group names", allow_empty=True
        )
        identities(self.graph_region, "graph region", allow_empty=True)
        declared = (
            self.direct_features
            + tuple(item.feature for item in self.weights)
            + tuple(feature for group in self.admissible_groups for feature in group.features)
            + self.graph_region
        )
        if not set(declared) <= set(self.feature_universe):
            raise ValueError("truth targets must belong to the feature universe")
        if not np.isfinite(self.center) or not np.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("truth center and scale must be finite with non-negative scale")
        signal = np.asarray(self.signal, dtype=np.float64)
        if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
            raise ValueError("materialized truth signal must be finite and one-dimensional")
        if self.kind == "null" and (np.any(signal != 0.0) or declared):
            raise ValueError("null truth cannot contain signal or recovery targets")
        copied = np.array(signal, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "signal", copied)


@dataclass(frozen=True, slots=True)
class SampledWeights:
    """Draw one weight per generating feature or group when the signal is built.

    Use it as `weights=` in `SparseTruth`, `CorrelatedTruth` or `ModuleTruth` to test recovery when
    generating features differ in importance. Each magnitude is drawn uniformly between `low` and
    `high`. The draw depends only on the seed and the number of weights, so the same seed gives
    the same weights, and it does not change which features an automatic selection picks. The
    drawn values are reported in the `weights` of the materialised truth.

    Attributes:
        low: Smallest magnitude, greater than zero.
        high: Largest magnitude, at least `low`. Equal bounds give every weight that magnitude.
        signs: `"positive"` keeps every weight positive. `"random"` gives each weight a positive
            or negative sign with equal probability.
    """

    low: float = 0.5
    high: float = 1.5
    signs: Literal["positive", "random"] = "positive"

    def __post_init__(self) -> None:
        if not (np.isfinite(self.low) and np.isfinite(self.high) and 0.0 < self.low <= self.high):
            raise ValueError("sampled weights require finite bounds with 0 < low <= high")
        if self.signs not in {"positive", "random"}:
            raise ValueError("sampled weight signs must be positive or random")


type WeightSpec = tuple[float, ...] | SampledWeights


@dataclass(frozen=True, slots=True)
class NullTruth:
    """A zero generating score."""


@dataclass(frozen=True, slots=True)
class SparseTruth:
    """Use a weighted sum of selected standardised features as the generating score.

    Specify either feature IDs or a count. Automatic selection searches in seeded random
    order and raises `ValueError` if it cannot find enough features within the correlation limit.

    Attributes:
        count: Number of features to pick automatically.
        features: Unique IDs of explicitly chosen features, also the exact selection targets.
        weights: Finite, non-zero coefficients in explicit feature order, defaulting to one, or
            `SampledWeights` to draw them, which also works with `count`.
        max_absolute_correlation: Maximum absolute Pearson correlation between any two automatically
            chosen features, in [0, 1). Ignored when `features` is supplied.
    """

    count: int | None = None
    features: tuple[str, ...] = ()
    weights: WeightSpec = ()
    max_absolute_correlation: float = 0.7

    def __post_init__(self) -> None:
        if (self.count is None) == (not self.features):
            raise ValueError("SparseTruth requires exactly one of count or features")
        if self.count is not None:
            integer(self.count, "sparse count", minimum=1)
        identities(self.features, "sparse features", allow_empty=True)
        if isinstance(self.weights, tuple):
            if self.weights and len(self.weights) != len(self.features):
                raise ValueError("sparse weights require one value per explicit feature")
            _validate_weights(self.weights)
        if not 0.0 <= self.max_absolute_correlation < 1.0:
            raise ValueError("max_absolute_correlation must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class CorrelatedTruth:
    """Define the generating score as a weighted sum of contributions from disjoint groups.

    A selection finds a group if it contains any member. The input data determines the correlations.

    Attributes:
        group_names: Names of the groups in `FeatureSets`, in coefficient order.
        sentinels: One generating feature per group, in `group_names` order. In `"sentinels"`
            mode an empty tuple picks one member per group using the seed. The generating features
            are the exact recovery targets in this mode. Leave empty in `"group_means"` mode.
        weights: Finite, non-zero coefficients in `group_names` order, defaulting to one, or
            `SampledWeights` to draw them.
        signal_form: `"sentinels"` uses the standardised value of each chosen feature.
            `"group_means"` uses the mean of each group's standardised features.
    """

    group_names: tuple[str, ...]
    sentinels: tuple[str, ...] = ()
    weights: WeightSpec = ()
    signal_form: Literal["sentinels", "group_means"] = "sentinels"

    def __post_init__(self) -> None:
        identities(self.group_names, "correlated group_names")
        identities(self.sentinels, "sentinels", allow_empty=True)
        if self.signal_form not in {"sentinels", "group_means"}:
            raise ValueError("signal_form must be sentinels or group_means")
        if self.signal_form == "group_means" and self.sentinels:
            raise ValueError("group_means has no direct sentinels")
        if self.sentinels and len(self.sentinels) != len(self.group_names):
            raise ValueError("sentinels must contain one feature per group")
        if isinstance(self.weights, tuple):
            if self.weights and len(self.weights) != len(self.group_names):
                raise ValueError("correlated weights require one value per group")
            _validate_weights(self.weights)


@dataclass(frozen=True, slots=True)
class ModuleTruth:
    """Define a generating score as a weighted sum over a named set of standardised features.

    Attributes:
        set_name: Name of the set in `FeatureSets`.
        weights: Finite, non-zero coefficients in set order, defaulting to one, or
            `SampledWeights` to draw them.

    Weighted recovery divides the sum of selected absolute weights by the total absolute weight.
    """

    set_name: str
    weights: WeightSpec = ()

    def __post_init__(self) -> None:
        identities((self.set_name,), "module set_name")
        if isinstance(self.weights, tuple):
            _validate_weights(self.weights)


@dataclass(frozen=True, slots=True)
class InteractionTruth:
    """Define a generating score that depends jointly on two standardised features.

    Attributes:
        features: Two distinct feature IDs that form the joint selection target.
        operation: `"product"` multiplies values. `"threshold"` tests if both exceed the cutoff.
        threshold: Finite cutoff in feature standard deviations. With `"threshold"`, the indicator
            is one when both standardised values exceed the cutoff and zero otherwise. The
            indicator is then standardised to form the generating score. A constant indicator raises
            `ValueError` during generation.
    """

    features: tuple[str, str]
    operation: Literal["product", "threshold"] = "product"
    threshold: float = 0.0

    def __post_init__(self) -> None:
        identities(self.features, "interaction features")
        if len(self.features) != 2:
            raise ValueError("interaction truth requires two distinct features")
        if self.operation not in {"product", "threshold"}:
            raise ValueError("operation must be product or threshold")
        if not np.isfinite(self.threshold):
            raise ValueError("interaction threshold must be finite")


@dataclass(frozen=True, slots=True)
class CustomSignal:
    """Supply a user-defined generating score and optional targets for feature selection.

    Attributes:
        signal: Finite, non-constant score with one value per input row.
        direct_features: Feature IDs counted as exact targets.
        groups: Groups credited when any member is selected.
        weights: Signed coefficients used to score coverage by absolute weight.
        graph_region: Target feature IDs for graph scoring.
        description: Optional label for the signal structure.

    All target feature IDs must occur in the input columns.
    """

    signal: NDArray[np.float64]
    direct_features: tuple[str, ...] = ()
    groups: tuple[RecoveryGroup, ...] = ()
    weights: tuple[FeatureWeight, ...] = ()
    graph_region: tuple[str, ...] = ()
    description: str = ""


type CustomTruthFunction = Callable[
    [NDArray[np.float64], tuple[str, ...], np.random.Generator], CustomSignal
]


@dataclass(frozen=True, slots=True)
class CustomTruth:
    """Define the generating score with a user-supplied function.

    Attributes:
        function: Callback taking a read-only matrix, column IDs and a seeded NumPy generator.

    The callback must return the same `CustomSignal` contents for the same inputs and seed.
    """

    function: CustomTruthFunction

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise ValueError("custom truth function must be callable")


type TruthSpec = (
    NullTruth | SparseTruth | CorrelatedTruth | ModuleTruth | InteractionTruth | CustomTruth
)


def _validate_weights(weights: tuple[float, ...]) -> None:
    if any(not np.isfinite(weight) or weight == 0.0 for weight in weights):
        raise ValueError("truth weights must be finite and non-zero")


_WEIGHT_STREAM = 0x5745494748  # keeps weight draws apart from automatic feature choices


def _resolve_weights(weights: WeightSpec, count: int, seed: int) -> NDArray[np.float64]:
    """Return one coefficient per generating feature or group, drawing them if requested."""
    if isinstance(weights, SampledWeights):
        rng = np.random.default_rng(np.random.SeedSequence((seed, _WEIGHT_STREAM)))
        drawn = rng.uniform(weights.low, weights.high, size=count)
        if weights.signs == "random":
            drawn = drawn * rng.choice(np.asarray([-1.0, 1.0]), size=count)
        return np.asarray(drawn, dtype=np.float64)
    return np.asarray(weights or tuple(1.0 for _ in range(count)), dtype=np.float64)


def _completed_standardized(
    population: Population, *, required_features: tuple[str, ...] = ()
) -> NDArray[np.float64]:
    """Median-complete observed columns and retain missing columns as constants."""
    required = _indices(population, required_features)
    matrix = np.array(population.X[:, required] if required_features else population.X, copy=True)
    all_missing = np.isnan(matrix).all(axis=0)
    unavailable = tuple(
        feature for feature, missing in zip(required_features, all_missing, strict=False) if missing
    )
    if unavailable:
        raise ValueError(f"truth features are entirely missing: {unavailable}")
    medians = np.zeros(matrix.shape[1], dtype=np.float64)
    observed = ~all_missing
    medians[observed] = np.nanmedian(matrix[:, observed], axis=0)
    missing_rows, missing_columns = np.where(np.isnan(matrix))
    matrix[missing_rows, missing_columns] = medians[missing_columns]
    scales = np.std(matrix, axis=0)
    constants = tuple(
        feature for feature, scale in zip(required_features, scales, strict=False) if scale == 0.0
    )
    if constants:
        raise ValueError(f"truth features are constant: {constants}")
    scales[scales == 0.0] = 1.0
    return np.asarray((matrix - np.mean(matrix, axis=0)) / scales, dtype=np.float64)


def _indices(population: Population, features: tuple[str, ...]) -> NDArray[np.int64]:
    positions = {feature: index for index, feature in enumerate(population.feature_ids)}
    unknown = tuple(feature for feature in features if feature not in positions)
    if unknown:
        raise ValueError(f"truth contains unknown feature identities: {unknown}")
    return np.asarray([positions[feature] for feature in features], dtype=np.int64)


def _standardize_signal(signal: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, float]:
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("truth signal must be finite and one-dimensional")
    center = float(np.mean(signal))
    scale = float(np.std(signal))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("a non-null truth signal must be nonconstant")
    return (signal - center) / scale, center, scale


def _weighted_signal(
    standardized: NDArray[np.float64], indices: NDArray[np.int64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    if not np.isfinite(weights).all() or np.all(weights == 0.0):
        raise ValueError("truth weights must be finite and not all zero")
    return np.asarray(standardized[:, indices] @ weights, dtype=np.float64)


def _select_sparse(
    standardized: NDArray[np.float64], count: int, maximum: float, rng: np.random.Generator
) -> NDArray[np.int64]:
    variable = np.flatnonzero(np.std(standardized, axis=0) > 0.0)
    selected: list[int] = []
    for candidate in rng.permutation(variable):
        if all(
            abs(float(np.corrcoef(standardized[:, candidate], standardized[:, prior])[0, 1]))
            <= maximum
            for prior in selected
        ):
            selected.append(int(candidate))
        if len(selected) == count:
            return np.asarray(selected, dtype=np.int64)
    raise ValueError(f"could not select {count} features under max_absolute_correlation={maximum}")


def _materialize_custom(truth: CustomTruth, population: Population, seed: int) -> MaterializedTruth:
    readonly = np.array(population.X, copy=True)
    readonly.setflags(write=False)
    first = truth.function(readonly, population.feature_ids, np.random.default_rng(seed))
    second = truth.function(readonly, population.feature_ids, np.random.default_rng(seed))
    first_signal = np.asarray(first.signal, dtype=np.float64)
    second_signal = np.asarray(second.signal, dtype=np.float64)
    if first_signal.shape != (population.sample_count,):
        raise ValueError("custom truth signal shape must equal the population row count")
    same_metadata = (
        first.direct_features == second.direct_features
        and first.groups == second.groups
        and first.weights == second.weights
        and first.graph_region == second.graph_region
        and first.description == second.description
    )
    if not np.array_equal(first_signal, second_signal, equal_nan=True) or not same_metadata:
        raise ValueError("custom truth must be deterministic for identical inputs and seed")
    declared = (
        first.direct_features
        + tuple(feature for group in first.groups for feature in group.features)
        + tuple(item.feature for item in first.weights)
        + first.graph_region
    )
    _indices(population, declared)
    signal, center, scale = _standardize_signal(first_signal)
    return MaterializedTruth(
        kind="custom",
        signal=signal,
        feature_universe=population.feature_ids,
        direct_features=first.direct_features,
        admissible_groups=first.groups,
        weights=first.weights,
        graph_region=first.graph_region,
        center=center,
        scale=scale,
        description=first.description,
    )


def materialize_truth(
    truth: TruthSpec,
    population: Population,
    *,
    seed: int,
    feature_sets: FeatureSets | None = None,
) -> MaterializedTruth:
    """Compute the generating score and record the feature-selection targets.

    Built-in specifications median-impute and standardise the feature columns they use.
    Non-null generating scores are standardised to mean zero and unit population standard
    deviation. Null scores are zero on every row.

    Args:
        truth: Built-in signal specification or a `CustomTruth` wrapping a callback.
        seed: Non-negative integer for automatic choices or the custom callback.
        feature_sets: Named feature groups required by correlated and module specifications.

    Raises:
        ValueError: Invalid settings or targets, an infeasible automatic selection, a non-finite
            or constant non-null generating score, or a custom callback that returns the wrong
            shape or differs between identical calls.
        TypeError: The specification type is unsupported.
    """
    integer(seed, "seed")
    if isinstance(truth, NullTruth):
        return MaterializedTruth(
            kind="null",
            signal=np.zeros(population.sample_count),
            feature_universe=population.feature_ids,
            center=0.0,
            scale=0.0,
            description="No biomarker signal",
        )

    rng = np.random.default_rng(seed)
    if isinstance(truth, SparseTruth):
        if truth.features:
            features = truth.features
            standardized = _completed_standardized(population, required_features=features)
            selected = np.arange(len(features), dtype=np.int64)
        else:
            if truth.count is None:
                raise RuntimeError("sparse truth count is missing")
            standardized = _completed_standardized(population)
            selected = _select_sparse(
                standardized, truth.count, truth.max_absolute_correlation, rng
            )
            features = tuple(population.feature_ids[index] for index in selected)
        weights = _resolve_weights(truth.weights, len(features), seed)
        raw = _weighted_signal(standardized, selected, weights)
        signal, center, scale = _standardize_signal(raw)
        return MaterializedTruth(
            kind="sparse",
            signal=signal,
            feature_universe=population.feature_ids,
            direct_features=features,
            weights=tuple(
                FeatureWeight(feature, float(weight))
                for feature, weight in zip(features, weights, strict=True)
            ),
            center=center,
            scale=scale,
            description=f"Sparse exact truth over {len(features)} features",
        )

    if isinstance(truth, CorrelatedTruth):
        if feature_sets is None:
            raise ValueError("CorrelatedTruth requires FeatureSets")
        groups = tuple(feature_sets.get(name) for name in truth.group_names)
        all_members = tuple(feature for group in groups for feature in group.members)
        if len(set(all_members)) != len(all_members):
            raise ValueError("correlated truth requires disjoint groups")
        for group in groups:
            _indices(population, group.members)
        sentinels = (
            (
                truth.sentinels
                or tuple(
                    group.members[int(rng.integers(0, len(group.members)))] for group in groups
                )
            )
            if truth.signal_form == "sentinels"
            else ()
        )
        if any(
            sentinel not in group.members
            for sentinel, group in zip(sentinels, groups, strict=False)
        ):
            raise ValueError("every correlated sentinel must belong to its declared group")
        required = (
            sentinels
            if truth.signal_form == "sentinels"
            else tuple(dict.fromkeys(feature for group in groups for feature in group.members))
        )
        standardized = _completed_standardized(population, required_features=required)
        positions = {feature: index for index, feature in enumerate(required)}
        weights = _resolve_weights(truth.weights, len(groups), seed)
        if truth.signal_form == "sentinels":
            selected = np.arange(len(sentinels), dtype=np.int64)
            raw = _weighted_signal(standardized, selected, weights)
            feature_weights = tuple(
                FeatureWeight(feature, float(weight))
                for feature, weight in zip(sentinels, weights, strict=True)
            )
        else:
            group_states = np.column_stack(
                [
                    np.mean(
                        standardized[:, [positions[feature] for feature in group.members]], axis=1
                    )
                    for group in groups
                ]
            )
            raw = np.asarray(group_states @ weights, dtype=np.float64)
            contributions: dict[str, float] = dict.fromkeys(required, 0.0)
            for group, weight in zip(groups, weights, strict=True):
                for feature in group.members:
                    contributions[feature] += float(weight) / len(group.members)
            feature_weights = tuple(
                FeatureWeight(feature, weight)
                for feature, weight in contributions.items()
                if weight != 0.0
            )
        signal, center, scale = _standardize_signal(raw)
        recovery_groups = tuple(
            RecoveryGroup(group.name, sentinels[index] if sentinels else None, group.members)
            for index, group in enumerate(groups)
        )
        return MaterializedTruth(
            kind="correlated",
            signal=signal,
            feature_universe=population.feature_ids,
            direct_features=sentinels,
            admissible_groups=recovery_groups,
            weights=feature_weights,
            center=center,
            scale=scale,
            description=f"Correlated {truth.signal_form} truth over {len(groups)} groups",
        )

    if isinstance(truth, ModuleTruth):
        if feature_sets is None:
            raise ValueError("ModuleTruth requires FeatureSets")
        feature_set = feature_sets.get(truth.set_name)
        standardized = _completed_standardized(population, required_features=feature_set.members)
        selected = np.arange(len(feature_set.members), dtype=np.int64)
        weights = _resolve_weights(truth.weights, len(feature_set.members), seed)
        if len(weights) != len(feature_set.members):
            raise ValueError("module weights require one value per set member")
        raw = _weighted_signal(standardized, selected, weights)
        signal, center, scale = _standardize_signal(raw)
        return MaterializedTruth(
            kind="module",
            signal=signal,
            feature_universe=population.feature_ids,
            weights=tuple(
                FeatureWeight(feature, float(weight))
                for feature, weight in zip(feature_set.members, weights, strict=True)
            ),
            center=center,
            scale=scale,
            description=f"Weighted module truth: {feature_set.name}",
        )

    if isinstance(truth, InteractionTruth):
        standardized = _completed_standardized(population, required_features=truth.features)
        left = standardized[:, 0]
        right = standardized[:, 1]
        if truth.operation == "product":
            raw = left * right
        else:
            raw = ((left > truth.threshold) & (right > truth.threshold)).astype(float)
        signal, center, scale = _standardize_signal(raw)
        return MaterializedTruth(
            kind="interaction",
            signal=signal,
            feature_universe=population.feature_ids,
            direct_features=truth.features,
            center=center,
            scale=scale,
            description=f"Pairwise {truth.operation} interaction",
        )

    if isinstance(truth, CustomTruth):
        return _materialize_custom(truth, population, seed)
    raise TypeError(f"unsupported truth specification: {type(truth).__name__}")
