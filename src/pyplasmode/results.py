"""Results returned by prediction, feature-recovery and stability evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RankingAtDepth:
    """Exact-target recovery for the leading features of a ranking.

    Attributes:
        depth: Cutoff in the ranking, counted from one.
        selected_count: Size of the top list.
        true_positive_count: Number of selected exact targets.
        target_count: Total exact targets.
        recall: Selected targets divided by all targets.
        precision: Selected targets divided by all selected features.
        false_discovery_fraction: Selected non-targets divided by all selected features.
    """

    depth: int
    selected_count: int
    true_positive_count: int
    target_count: int
    recall: float
    precision: float
    false_discovery_fraction: float


@dataclass(frozen=True, slots=True)
class RankingEvaluation:
    """Results for finding exact target features in a ranking.

    Attributes:
        first_relevant_rank: One-based position of the first exact target. `evaluate_ranking`
            always sets it because it requires targets and a complete ranking.
        at_depth: Counts, precision, recall and false discovery fraction in depth order.
    """

    first_relevant_rank: int | None
    at_depth: tuple[RankingAtDepth, ...]


@dataclass(frozen=True, slots=True)
class GroupAtDepth:
    """Group recovery at one depth.

    Attributes:
        depth: Number of leading features selected.
        recovered_group_count: Number of groups with at least one member selected.
        group_count: Total target groups.
        group_recall: Groups with a selected member divided by all target groups.
    """

    depth: int
    recovered_group_count: int
    group_count: int
    group_recall: float


@dataclass(frozen=True, slots=True)
class GroupEvaluation:
    """Group recovery across depths.

    Attributes:
        at_depth: Group counts and recall by cutoff.
    """

    at_depth: tuple[GroupAtDepth, ...]


@dataclass(frozen=True, slots=True)
class ModuleAtDepth:
    """Weight recovered by the leading features of a ranking.

    Attributes:
        depth: Number of leading features selected.
        recovered_member_count: Number of selected weighted features.
        member_count: Total weighted features.
        weighted_coverage: Sum of selected absolute weights divided by the total absolute weight.
    """

    depth: int
    recovered_member_count: int
    member_count: int
    weighted_coverage: float


@dataclass(frozen=True, slots=True)
class ModuleEvaluation:
    """Weighted recovery across depths.

    Attributes:
        at_depth: Counts and absolute-weight coverage by cutoff.
    """

    at_depth: tuple[ModuleAtDepth, ...]


@dataclass(frozen=True, slots=True)
class GraphRegionAtDepth:
    """Recovery of a graph region by the first `depth` features of a ranking.

    Attributes:
        depth: Number of leading features counted as selected.
        node_coverage: Fraction of the region's nodes that are selected.
        edge_coverage: Fraction of the region's edges with both ends selected, or `None`
            when the region has no edges.
        target_edge_count: Number of edges joining two nodes of the region.
    """

    depth: int
    node_coverage: float
    edge_coverage: float | None
    target_edge_count: int = 0


@dataclass(frozen=True, slots=True)
class GraphRegionEvaluation:
    """Graph region recovery across depths.

    Attributes:
        at_depth: Node and edge coverage by cutoff.
    """

    at_depth: tuple[GraphRegionAtDepth, ...]


@dataclass(frozen=True, slots=True)
class SignalReconstructionEvaluation:
    """Held-out performance of a linear model that predicts the generating score from a panel.

    Attributes:
        selected_feature_count: Number of features supplied to the model.
        evaluation_count: Number of held-out rows.
        r_squared: Held-out R-squared.
        root_mean_squared_error: Root mean squared prediction error in generating-score units.
    """

    selected_feature_count: int
    evaluation_count: int
    r_squared: float
    root_mean_squared_error: float


@dataclass(frozen=True, slots=True)
class PredictionEvaluation:
    """Metric name, metric value and number of observations scored."""

    metric: str
    value: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class PanelEvaluation:
    """Prediction scores for a panel and a reference model.

    Attributes:
        metric: Name of the scoring rule.
        reference_value: Metric value for the reference model.
        panel_value: Metric value for the panel.
        difference: Panel value minus reference value. Negative favours the panel for loss metrics.
        sample_count: Number of shared evaluation rows.
    """

    metric: str
    reference_value: float
    panel_value: float
    difference: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class RankStabilityEvaluation:
    """Agreement averaged over pairs of feature rankings.

    Attributes:
        ranking_count: Number of rankings.
        feature_count: Features per ranking.
        depth: Features per top list.
        mean_leading_jaccard: Mean fraction of the two top lists' union shared by both lists.
        mean_leading_intersection: Mean number of features shared by the two top lists.
        expected_random_intersection: Expected shared feature count for independent random rankings
            (`depth**2 / feature_count`).
        excess_leading_intersection: Mean shared count minus the random expectation.
        mean_spearman: Mean pairwise Spearman correlation.
        mean_rank_biased_overlap: Mean pairwise rank overlap, weighted towards the top.
    """

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
    """Mean and uncertainty across independent successful simulations.

    Attributes:
        replicate_count: Number of successful simulations.
        failure_count: Number of failed simulations.
        mean: Mean of successful estimates.
        standard_deviation: Sample standard deviation of successful estimates.
        standard_error: Sample standard deviation divided by the square root of the success count.
        confidence_interval_low: Lower bound of the 95% t interval for the mean.
        confidence_interval_high: Upper bound of that interval.
    """

    replicate_count: int
    failure_count: int
    mean: float
    standard_deviation: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float


@dataclass(frozen=True, slots=True)
class PairedReplicateComparison:
    """Differences between two procedures evaluated on the same simulated datasets.

    Attributes:
        replicate_ids: Shared dataset IDs for which both procedures succeeded, in left input order.
        differences: Right minus left for each ID in `replicate_ids`.
        matched_pair_count: Shared dataset count, including failures.
        failed_pair_count: Shared datasets with a failure on either side.
        left_failure_count: Shared datasets where the left estimate failed.
        right_failure_count: Shared datasets where the right estimate failed.
        unmatched_left_count: IDs supplied only on the left.
        unmatched_right_count: IDs supplied only on the right.
        summary: Mean and 95% t interval across jointly successful pairs.
    """

    replicate_ids: tuple[str, ...]
    differences: tuple[float, ...]
    matched_pair_count: int
    failed_pair_count: int
    left_failure_count: int
    right_failure_count: int
    unmatched_left_count: int
    unmatched_right_count: int
    summary: ReplicateSummary


@dataclass(frozen=True, slots=True)
class FractionalRecovery:
    """Recovery averaged over ties at the selection cutoff.

    Attributes:
        estimand: Recovery measure for exact targets, groups, weights or interactions.
        value: Score in [0, 1], with the denominator or event defined by `estimand`.
        truth_support: Count of targets for `estimand` (two for an interaction).
        positive_score_support_size: Number of positive scores, including those below the cutoff.
        requested_depth: Requested slot count.
        achieved_size: Features with positive selection probability.

    Exact recovery averages selection probability over targets. Group recovery averages the
    chance of any hit over groups. Module recovery divides expected selected absolute weight
    by total absolute weight. Interaction recovery is the chance of selecting both targets.
    """

    estimand: Literal["exact", "group", "module", "interaction"]
    value: float
    truth_support: int
    positive_score_support_size: int
    requested_depth: int
    achieved_size: int


@dataclass(frozen=True, slots=True)
class MatchedRecovery:
    """Feature recovery compared with random selection within matching groups.

    Attributes:
        estimand: Recovery measure for exact targets, groups, weights or interactions.
        method: `"analytic"`, giving the chance expectation by calculation.
        requested_depth: Requested slot count.
        achieved_size: Features with positive selection probability.
        observed_recovery: Recovery score for the supplied selection.
        expected_recovery: Mean recovery score under matched random selection.
        chance_adjusted_recovery: Recovery above chance (observed minus expected recovery).
        admissible_panel_count: Number of ways to assign positive-score status and selection
            probabilities within matching groups, counting assignments separately even if they
            can yield the same final panel.
    """

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
    """Recovery above chance for the outcome's targets compared with alternatives.

    Attributes:
        estimand: Recovery measure for exact targets, groups, weights or interactions.
        own_adjusted_recovery: Recovery above chance for the outcome's targets.
        mean_foreign_adjusted_recovery: Mean recovery above chance across alternatives.
        specificity: Own adjusted recovery minus the alternative mean.
        foreign_truth_count: Number of alternative target sets.
    """

    estimand: Literal["exact", "group", "module", "interaction"]
    own_adjusted_recovery: float
    mean_foreign_adjusted_recovery: float
    specificity: float
    foreign_truth_count: int
