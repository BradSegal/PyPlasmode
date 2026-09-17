"""Pure biomarker recovery, prediction and stability evaluation."""

from __future__ import annotations

from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr, t
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import make_pipeline

from pyplasmode._validation import indices as integer_indices
from pyplasmode._validation import integer
from pyplasmode.results import (
    GraphRegionAtDepth,
    GraphRegionEvaluation,
    GroupAtDepth,
    GroupEvaluation,
    ModuleAtDepth,
    ModuleEvaluation,
    PanelEvaluation,
    PredictionEvaluation,
    RankingAtDepth,
    RankingEvaluation,
    RankStabilityEvaluation,
    ReplicateSummary,
    SignalReconstructionEvaluation,
)
from pyplasmode.structures import FeatureGraph
from pyplasmode.truth import MaterializedTruth

type PredictionMetric = Literal["roc_auc", "average_precision", "log_loss", "r2", "rmse"]


def _ranking(ranking: tuple[str, ...], universe: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking contains duplicate feature identities")
    universe_set = set(universe)
    unknown = tuple(feature for feature in ranking if feature not in universe_set)
    if unknown:
        raise ValueError(f"ranking contains unknown feature identities: {unknown}")
    if len(ranking) != len(universe) or set(ranking) != set(universe):
        raise ValueError("ranking must be a complete permutation of the feature universe")
    return tuple(ranking)


def _depths(depths: tuple[int, ...], feature_count: int) -> tuple[int, ...]:
    for depth in depths:
        integer(depth, "depth", minimum=1)
    if not depths or tuple(sorted(set(depths))) != depths:
        raise ValueError("depths must be non-empty, unique and strictly increasing")
    if any(isinstance(depth, bool) or depth <= 0 or depth > feature_count for depth in depths):
        raise ValueError("each depth must be between one and the feature count")
    return depths


def evaluate_ranking(
    ranking: tuple[str, ...],
    truth: MaterializedTruth,
    *,
    depths: tuple[int, ...],
) -> RankingEvaluation:
    """Evaluate exact generating-feature recovery at declared leading depths."""
    ordered = _ranking(ranking, truth.feature_universe)
    checked_depths = _depths(depths, len(ordered))
    target = set(truth.direct_features)
    if not target:
        raise ValueError("this truth has no exact generating-feature target")
    first = min(
        (index + 1 for index, feature in enumerate(ordered) if feature in target),
        default=None,
    )
    rows: list[RankingAtDepth] = []
    for depth in checked_depths:
        selected = set(ordered[:depth])
        true_positives = len(selected & target)
        precision = true_positives / len(selected)
        rows.append(
            RankingAtDepth(
                depth=depth,
                selected_count=len(selected),
                true_positive_count=true_positives,
                target_count=len(target),
                recall=true_positives / len(target),
                precision=precision,
                false_discovery_fraction=1.0 - precision,
            )
        )
    return RankingEvaluation(first, tuple(rows))


def evaluate_groups(
    ranking: tuple[str, ...],
    truth: MaterializedTruth,
    *,
    depths: tuple[int, ...],
) -> GroupEvaluation:
    """Evaluate recovery of at least one admissible feature per declared group."""
    ordered = _ranking(ranking, truth.feature_universe)
    checked_depths = _depths(depths, len(ordered))
    groups = truth.admissible_groups
    if not groups:
        raise ValueError("this truth has no admissible substitute groups")
    rows = []
    for depth in checked_depths:
        selected = set(ordered[:depth])
        recovered = sum(bool(selected & set(group.features)) for group in groups)
        rows.append(GroupAtDepth(depth, recovered, len(groups), recovered / len(groups)))
    return GroupEvaluation(tuple(rows))


def evaluate_module(
    ranking: tuple[str, ...],
    truth: MaterializedTruth,
    *,
    depths: tuple[int, ...],
) -> ModuleEvaluation:
    """Evaluate absolute-weight coverage of a distributed truth module."""
    ordered = _ranking(ranking, truth.feature_universe)
    checked_depths = _depths(depths, len(ordered))
    weights = {item.feature: abs(item.weight) for item in truth.weights}
    if not weights:
        raise ValueError("this truth has no weighted module target")
    total = sum(weights.values())
    rows = []
    for depth in checked_depths:
        selected = set(ordered[:depth])
        recovered = selected & set(weights)
        rows.append(
            ModuleAtDepth(
                depth,
                len(recovered),
                len(weights),
                sum(weights[feature] for feature in recovered) / total,
            )
        )
    return ModuleEvaluation(tuple(rows))


def evaluate_graph_region(
    ranking: tuple[str, ...],
    graph: FeatureGraph,
    *,
    region: tuple[str, ...],
    depths: tuple[int, ...],
) -> GraphRegionEvaluation:
    """Evaluate node and induced-edge recovery for a declared graph region."""
    ordered = _ranking(ranking, graph.nodes)
    checked_depths = _depths(depths, len(ordered))
    region_set = set(region)
    if not region_set or not region_set <= set(graph.nodes):
        raise ValueError("graph region must be a non-empty subset of graph nodes")
    region_edges = {
        frozenset((edge.source, edge.target))
        for edge in graph.edges
        if edge.source in region_set and edge.target in region_set
    }
    rows = []
    for depth in checked_depths:
        selected = set(ordered[:depth])
        recovered_edges = sum(edge <= selected for edge in region_edges)
        edge_coverage = recovered_edges / len(region_edges) if region_edges else 0.0
        rows.append(
            GraphRegionAtDepth(
                depth,
                len(selected & region_set) / len(region_set),
                edge_coverage,
            )
        )
    return GraphRegionEvaluation(tuple(rows))


def _indices(indices: ArrayLike, size: int, name: str) -> NDArray[np.int64]:
    values = integer_indices(indices, name)
    if values.ndim != 1 or not values.size or np.any((values < 0) | (values >= size)):
        raise ValueError(f"{name} must contain valid row indices")
    if len(set(values.tolist())) != values.size:
        raise ValueError(f"{name} must not contain duplicates")
    return values


def evaluate_signal_reconstruction(
    X: ArrayLike,
    feature_ids: tuple[str, ...],
    *,
    selected_features: tuple[str, ...],
    signal: ArrayLike,
    development_indices: ArrayLike,
    evaluation_indices: ArrayLike,
) -> SignalReconstructionEvaluation:
    """Fit a maintained linear model and evaluate held-out latent-signal reconstruction."""
    matrix = np.asarray(X, dtype=np.float64)
    target = np.asarray(signal, dtype=np.float64)
    if matrix.ndim != 2 or target.shape != (matrix.shape[0],):
        raise ValueError("X and signal dimensions are incompatible")
    if len(feature_ids) != matrix.shape[1] or len(set(feature_ids)) != len(feature_ids):
        raise ValueError("feature_ids must uniquely identify every X column")
    if len(set(selected_features)) != len(selected_features):
        raise ValueError("selected_features must be unique")
    positions = {feature: index for index, feature in enumerate(feature_ids)}
    unknown = tuple(feature for feature in selected_features if feature not in positions)
    if unknown:
        raise ValueError(f"selected_features contain unknown identities: {unknown}")
    columns = [positions[feature] for feature in selected_features]
    selected = matrix[:, columns] if columns else np.zeros((matrix.shape[0], 1), dtype=np.float64)
    if np.isinf(selected).any() or not np.isfinite(target).all():
        raise ValueError("signal reconstruction contains infinite values or a non-finite signal")
    development = _indices(development_indices, matrix.shape[0], "development_indices")
    evaluation = _indices(evaluation_indices, matrix.shape[0], "evaluation_indices")
    if set(development.tolist()) & set(evaluation.tolist()):
        raise ValueError("development and evaluation indices must be disjoint")
    model = (
        make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
        if columns
        else DummyRegressor(strategy="mean")
    ).fit(selected[development], target[development])
    predictions = model.predict(selected[evaluation])
    return SignalReconstructionEvaluation(
        len(columns),
        evaluation.size,
        evaluate_prediction(target[evaluation], predictions, metric="r2").value,
        evaluate_prediction(target[evaluation], predictions, metric="rmse").value,
    )


def evaluate_prediction(
    target: ArrayLike, prediction: ArrayLike, *, metric: PredictionMetric
) -> PredictionEvaluation:
    """Evaluate supplied predictions with one maintained standard metric."""
    observed = np.asarray(target, dtype=np.float64)
    estimated = np.asarray(prediction, dtype=np.float64)
    if (
        observed.ndim != 1
        or not observed.size
        or estimated.shape != observed.shape
        or not np.isfinite(observed).all()
        or not np.isfinite(estimated).all()
    ):
        raise ValueError("target and prediction must be finite aligned one-dimensional arrays")
    if (
        metric in {"roc_auc", "average_precision", "log_loss"}
        and not np.isin(observed, (0.0, 1.0)).all()
    ):
        raise ValueError("classification targets must be binary zero/one labels")
    if metric == "roc_auc":
        if np.unique(observed).size != 2:
            raise ValueError("ROC AUC requires both target classes")
        value = roc_auc_score(observed, estimated)
    elif metric == "average_precision":
        if not np.any(observed == 1.0):
            raise ValueError("average precision requires a positive target")
        value = average_precision_score(observed, estimated)
    elif metric == "log_loss":
        if np.any((estimated < 0.0) | (estimated > 1.0)):
            raise ValueError("log loss requires probabilities in [0, 1]")
        value = log_loss(observed, estimated, labels=[0, 1])
    elif metric == "r2":
        if observed.size < 2 or np.ptp(observed) == 0.0:
            raise ValueError("R-squared requires at least two observations and a varying target")
        value = r2_score(observed, estimated)
    elif metric == "rmse":
        value = root_mean_squared_error(observed, estimated)
    else:
        raise ValueError(f"unsupported prediction metric: {metric}")
    if not np.isfinite(value):
        raise ValueError(f"{metric} is not finite for the supplied observations")
    return PredictionEvaluation(metric, float(value), observed.size)


def evaluate_panel(
    target: ArrayLike,
    reference_prediction: ArrayLike,
    panel_prediction: ArrayLike,
    *,
    metric: PredictionMetric,
) -> PanelEvaluation:
    """Compare a panel's supplied predictions with reference-model predictions."""
    reference = evaluate_prediction(target, reference_prediction, metric=metric)
    panel = evaluate_prediction(target, panel_prediction, metric=metric)
    return PanelEvaluation(
        metric,
        reference.value,
        panel.value,
        panel.value - reference.value,
        reference.sample_count,
    )


def _rank_positions(ranking: tuple[str, ...]) -> dict[str, int]:
    return {feature: index for index, feature in enumerate(ranking, start=1)}


def _rbo(left: tuple[str, ...], right: tuple[str, ...], persistence: float) -> float:
    overlap = 0
    weighted = 0.0
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    for depth, (left_item, right_item) in enumerate(zip(left, right, strict=True), start=1):
        overlap += int(left_item in right_seen)
        left_seen.add(left_item)
        overlap += int(right_item in left_seen)
        right_seen.add(right_item)
        agreement = overlap / depth
        weighted += agreement * persistence ** (depth - 1)
    return float((1.0 - persistence) * weighted + (overlap / len(left)) * persistence ** len(left))


def rank_stability(
    rankings: tuple[tuple[str, ...], ...],
    *,
    depth: int,
    persistence: float = 0.9,
) -> RankStabilityEvaluation:
    """Summarize pairwise leading-set, Spearman and rank-biased overlap stability."""
    if len(rankings) < 2:
        raise ValueError("rank_stability requires at least two rankings")
    universe = tuple(rankings[0])
    if len(universe) < 2:
        raise ValueError("rank_stability requires at least two features for Spearman agreement")
    checked = tuple(_ranking(ranking, universe) for ranking in rankings)
    _depths((depth,), len(universe))
    if not 0.0 < persistence < 1.0:
        raise ValueError("persistence must be between zero and one")
    jaccards: list[float] = []
    intersections: list[float] = []
    correlations: list[float] = []
    rbos: list[float] = []
    for left, right in combinations(checked, 2):
        left_top = set(left[:depth])
        right_top = set(right[:depth])
        intersection = len(left_top & right_top)
        intersections.append(float(intersection))
        jaccards.append(intersection / len(left_top | right_top))
        left_positions = _rank_positions(left)
        right_positions = _rank_positions(right)
        correlations.append(
            float(
                spearmanr(
                    [left_positions[feature] for feature in universe],
                    [right_positions[feature] for feature in universe],
                ).statistic
            )
        )
        rbos.append(_rbo(left, right, persistence))
    expected = depth**2 / len(universe)
    mean_intersection = float(np.mean(intersections))
    return RankStabilityEvaluation(
        len(checked),
        len(universe),
        depth,
        float(np.mean(jaccards)),
        mean_intersection,
        expected,
        mean_intersection - expected,
        float(np.mean(correlations)),
        float(np.mean(rbos)),
    )


def summarize_replicates(values: ArrayLike, *, failure_count: int = 0) -> ReplicateSummary:
    """Summarize replicate estimates with a 95% Monte Carlo t interval."""
    estimates = np.asarray(values, dtype=np.float64)
    if estimates.ndim != 1 or estimates.size < 2 or not np.isfinite(estimates).all():
        raise ValueError("uncertainty requires at least two finite replicate values")
    if isinstance(failure_count, bool) or not isinstance(failure_count, int) or failure_count < 0:
        raise ValueError("failure_count must be a non-negative integer")
    mean = float(np.mean(estimates))
    standard_deviation = float(np.std(estimates, ddof=1))
    standard_error = standard_deviation / np.sqrt(estimates.size)
    critical = float(t.ppf(0.975, df=estimates.size - 1))
    low = mean - critical * standard_error
    high = mean + critical * standard_error
    return ReplicateSummary(
        estimates.size,
        failure_count,
        mean,
        standard_deviation,
        standard_error,
        low,
        high,
    )
