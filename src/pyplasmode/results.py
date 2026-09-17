"""Typed aggregate results returned by biomarker evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RankingAtDepth:
    """Exact-feature recovery at one leading-list depth."""

    depth: int
    selected_count: int
    true_positive_count: int
    target_count: int
    recall: float
    precision: float
    false_discovery_fraction: float


@dataclass(frozen=True, slots=True)
class RankingEvaluation:
    """Exact recovery across declared depths."""

    first_relevant_rank: int | None
    at_depth: tuple[RankingAtDepth, ...]


@dataclass(frozen=True, slots=True)
class GroupAtDepth:
    """Admissible substitute-group recovery at one depth."""

    depth: int
    recovered_group_count: int
    group_count: int
    group_recall: float


@dataclass(frozen=True, slots=True)
class GroupEvaluation:
    """Recovery of at least one declared member per substitute group."""

    at_depth: tuple[GroupAtDepth, ...]


@dataclass(frozen=True, slots=True)
class ModuleAtDepth:
    """Absolute-weight coverage of a distributed module at one depth."""

    depth: int
    recovered_member_count: int
    member_count: int
    weighted_coverage: float


@dataclass(frozen=True, slots=True)
class ModuleEvaluation:
    """Distributed module coverage across declared depths."""

    at_depth: tuple[ModuleAtDepth, ...]


@dataclass(frozen=True, slots=True)
class GraphRegionAtDepth:
    """Node and induced-edge recovery for a declared graph region."""

    depth: int
    node_coverage: float
    edge_coverage: float


@dataclass(frozen=True, slots=True)
class GraphRegionEvaluation:
    """Graph-region recovery across declared depths."""

    at_depth: tuple[GraphRegionAtDepth, ...]


@dataclass(frozen=True, slots=True)
class SignalReconstructionEvaluation:
    """Held-out linear reconstruction of a known latent signal."""

    selected_feature_count: int
    evaluation_count: int
    r_squared: float
    root_mean_squared_error: float


@dataclass(frozen=True, slots=True)
class PredictionEvaluation:
    """One standard prediction metric over supplied outputs."""

    metric: str
    value: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class PanelEvaluation:
    """Prediction retained by a panel relative to supplied reference predictions."""

    metric: str
    reference_value: float
    panel_value: float
    difference: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class RankStabilityEvaluation:
    """Pairwise leading-set, full-rank and top-weighted stability."""

    ranking_count: int
    feature_count: int
    depth: int
    mean_leading_jaccard: float
    mean_leading_intersection: float
    expected_random_intersection: float
    excess_leading_intersection: float
    mean_spearman: float
    mean_rank_biased_overlap: float


@dataclass(frozen=True, slots=True)
class ReplicateSummary:
    """Mean, Monte Carlo uncertainty and failure count across replicates."""

    replicate_count: int
    failure_count: int
    mean: float
    standard_deviation: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float


@dataclass(frozen=True, slots=True)
class FractionalRecovery:
    """Mechanism-specific recovery for one fractional nomination."""

    estimand: Literal["exact", "group", "module", "interaction"]
    value: float
    truth_support: int
    positive_score_support_size: int
    requested_depth: int
    achieved_size: int


@dataclass(frozen=True, slots=True)
class MatchedRecovery:
    """Observed recovery calibrated against an analytic matched reference."""

    estimand: Literal["exact", "group", "module", "interaction"]
    method: Literal["analytic"]
    requested_depth: int
    achieved_size: int
    observed_recovery: float
    expected_recovery: float
    chance_adjusted_recovery: float
    admissible_panel_count: int


@dataclass(frozen=True, slots=True)
class OutcomeSpecificity:
    """Own-truth recovery relative to disjoint foreign truths."""

    estimand: Literal["exact", "group", "module", "interaction"]
    own_adjusted_recovery: float
    mean_foreign_adjusted_recovery: float
    specificity: float
    foreign_truth_count: int
