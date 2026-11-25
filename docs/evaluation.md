# Evaluation

Prediction metrics describe the model's accuracy; recovery metrics compare its selected
features with the known signal. A *panel* is a selected subset of features used to predict.
Compute each method's results on the same simulated dataset, one independent draw of outcomes
from a design, to make the comparison paired.

| Question | Function |
| --- | --- |
| Which generating features were found? | `evaluate_ranking` |
| Was at least one member of each relevant group found? | `evaluate_groups` |
| How much generating weight does the panel cover? | `evaluate_module` |
| What fraction of a graph region's nodes and edges was recovered? | `evaluate_graph_region` |
| Can the panel reconstruct the generating score? | `evaluate_signal_reconstruction` |
| How accurate are the predictions? | `evaluate_prediction` |
| How does a smaller panel compare with a reference model? | `evaluate_panel` |
| How similar are repeated rankings? | `rank_stability` |
| What is the average result across simulations? | `summarize_replicates` |
| Which of two procedures does better across simulations? | `compare_paired_replicates` |
| How is recovery shared when importance scores tie? | `evaluate_fractional_recovery` |
| How far is recovery above chance? | `evaluate_matched_recovery` |
| Does a selection prefer its own signal to other signals? | `evaluate_outcome_specificity` |

[Metrics](metrics.md) covers `pyplasmode.metrics` functions for selection stability, rank
agreement with ties, consensus rankings, AUC under censoring, bootstrap differences and the
discrimination gap.

## Feature and group recovery

Recovery is the fraction of the generating features, their groups or their weight that a
selection contains. The ranking evaluators take a ranking: a tuple containing every feature
once, best first. For each `k` in `depths` they score the first `k` features. If your scores
tie, break the ties yourself or use [tied scores](#rankings-and-scores) instead.

Suppose A and B generate the outcome, A with nine times B's weight, and a model ranks A and C
first.

```python
import numpy as np
import pyplasmode as ppm

population = ppm.Population(np.random.default_rng(7).normal(size=(100, 4)), ("A", "B", "C", "D"))
truth = ppm.materialize_truth(
    ppm.SparseTruth(features=("A", "B"), weights=(9.0, 1.0)), population, seed=3
)
ranking = ("A", "C", "B", "D")
result = ppm.evaluate_ranking(ranking, truth, depths=(2,))
print(result.at_depth[0].recall)  # 0.5: A of A and B
print(result.at_depth[0].precision)  # 0.5: A of A and C
module = ppm.evaluate_module(ranking, truth, depths=(2,))
print(module.at_depth[0].weighted_coverage)  # 0.9: weight 9 of 9 + 1
```

Exact recovery is the `recall` field. For group recovery, suppose the signal declares groups
{A} and {B, C}, with C a correlate of B: a measured feature strongly correlated with it.

| Question | Example | Score | Function and field |
| --- | --- | --- | --- |
| Did we find the generating features? | A and B generate risk; A and C are selected. | Exact recovery 1/2 | `evaluate_ranking`, `recall` |
| Are the selections generating features? | One of two selected features generates risk. | Precision 1/2 | `evaluate_ranking`, `precision` |
| Did we find their groups? | Groups are {A} and {B, C}; A and C are selected. | Group recovery 2/2 | `evaluate_groups`, `group_recall` |
| How much generating weight is covered? | A has weight 9 and B weight 1; only A is selected. | Weighted recovery 9/10 | `evaluate_module`, `weighted_coverage` |
| Was an interaction recovered? | The signal requires A and B; only A is selected. | Interaction recovery 0 | `evaluate_fractional_recovery`, `value` |

The [worked recovery script](https://github.com/BradSegal/PyPlasmode/blob/main/examples/recovery.py)
runs the exact, weighted and group examples.

Let `S_k` be the first `k` features of the ranking and `T` the set of generating features
(`truth.direct_features`). Exact recovery is `|S_k intersect T| / |T|`; precision is
`|S_k intersect T| / k`; `false_discovery_fraction` is `1 - precision`. Exact recovery
requires a non-empty `T`; use group or weighted recovery for distributed signals. Weighted
recovery is the sum of absolute generating weights of the selected features divided by the
sum over all generating features. Interaction recovery is 1 when both features of the
interaction are selected and 0 otherwise. No ranking evaluator computes it: pass the
selection to `evaluate_fractional_recovery`.

## Rankings and scores

`nomination_from_scores(feature_ids, scores, depth=...)` selects `depth` features from
non-negative importance scores without breaking ties arbitrarily. It returns a
`FractionalNomination` (a selection is a *nomination* in the API) giving each feature a
membership: the probability that it is selected when tied slots are filled at random. The
cutoff is the `depth`-th highest positive score.

- A score above the cutoff has membership 1.
- Scores tied at the cutoff share the remaining slots equally.
- Lower and zero scores have membership 0. With fewer than `depth` positive scores, the
  selection is smaller than `depth`.

`evaluate_fractional_recovery(nomination, truth, estimand=...)` returns the recovery in
`value`, with `estimand` one of `"exact"`, `"group"`, `"module"` (weighted recovery) or
`"interaction"`. Here B, C and D tie for the two slots left after A at `depth=3`:

```python
tied = ppm.nomination_from_scores(("A", "B", "C", "D"), (5.0, 2.0, 2.0, 2.0), depth=3)
print(tied.membership.round(3))  # [1.    0.667 0.667 0.667]: two slots for three features
print(tied.selected_features, tied.achieved_size)  # ('A', 'B', 'C', 'D') 4
print(round(tied.membership_budget, 3))  # 3.0
for estimand in ("exact", "module", "interaction"):
    recovery = ppm.evaluate_fractional_recovery(tied, truth, estimand=estimand)
    print(estimand, round(recovery.value, 3))
# exact 0.833: (1 + 2/3) / 2
# module 0.967: (9 x 1 + 1 x 2/3) / 10
# interaction 0.667: A is certain and B is chosen with probability 2/3
```

`selected_features` lists the features with non-zero membership and `achieved_size` counts
them, so it exceeds `depth` when more features tie than slots remain. `membership_budget`
sums the memberships: `depth`, or the number of positive scores if smaller.

Exact and weighted recovery replace 0 or 1 by the membership `m_j`: exact recovery is
`sum(j in T, m_j) / |T|` and weighted recovery `sum(j, |w_j| m_j) / sum(j, |w_j|)`, with
`w_j` the generating weight of feature `j`. Group and interaction recovery count the equally
likely ways to fill the tied slots. Let `b` be the number of tied features, `r` the slots
left for them and `t` the number of target features among them:

- Group recovery, the probability of selecting at least one member, is 1 if a member has
  membership 1 and otherwise `1 - choose(b-t, r) / choose(b, r)`: one minus the share of
  fillings that miss the group.
- Interaction recovery, the probability of selecting both features, is 0 if either has
  membership 0 or `r < t`, and otherwise `choose(b-t, r-t) / choose(b, r)`: the share of
  fillings that contain every tied one.

In the example `b = 3`, `r = 2` and the fillings are {B, C}, {B, D} and {C, D}. For the
interaction of A and B, only B is tied (`t = 1`): `choose(2, 1) / choose(3, 2) = 2/3`. A
group {B, C} (`t = 2`) gives `1 - choose(1, 2) / choose(3, 2) = 1`: every filling contains
B or C. An interaction of B and C would give `choose(1, 0) / choose(3, 2) = 1/3`, not the
product of memberships 4/9.

Group recovery needs a truth that declares groups and interaction recovery one with exactly
two generating features; otherwise the call raises `ValueError`. For a nomination over a
subset of the truth's features, pass `allow_subset=True`: features outside it get no credit.

## Matched chance recovery

Selecting ten out of 100 exchangeable features recovers 10% of the generating features
on average, even if the ranking contains no information. Recovering 40% is therefore
30 percentage points above chance. Random selection is more likely to hit a larger group,
so a matched reference also accounts for group size.

`evaluate_matched_recovery` returns recovery above chance, the observed recovery minus its
expectation under random feature assignment (`observed_recovery`, `expected_recovery`, and
`chance_adjusted_recovery`). Supply strata of features with similar properties, defined without
reference to the signal or evaluation outcomes. Within each stratum the reference reassigns
feature identities while preserving the positive-score count and selection membership weights.

The [tutorial](tutorial.md#compare-with-matched-chance) builds strata for correlated pairs
and independent features. For empirical inputs, its
[correlation-bin example](tutorial.md#matching-strata-on-empirical-measurements)
constructs strata from measurements; align the resulting tuple with the selection's
feature order before calling `evaluate_matched_recovery`.

Expectations are computed analytically for exact, group, weighted and interaction recovery.
For exact recovery, expected membership of a feature in stratum `h` is
`sum(j in h, m_j) / |h|`. If only one assignment is possible, observed and expected
recovery coincide and recovery above chance is zero.

### Outcome specificity

A method can recover generating features without using the outcome, for example by
favouring high-variance features or hubs correlated with many others. It then also recovers
signals it never saw. `evaluate_outcome_specificity` tests for this by scoring one selection
against `own_truth`, the signal behind the outcome the model was fitted to, and
`foreign_truths`, a tuple of other signals on the same features.

It computes recovery above chance for each signal with `evaluate_matched_recovery` and
returns `own_adjusted_recovery`, the mean for the other signals
(`mean_foreign_adjusted_recovery`) and `specificity`, the first minus the second:

```python
other_truth = ppm.materialize_truth(ppm.SparseTruth(features=("C", "D")), population, seed=3)
top_two = ppm.nomination_from_scores(("A", "B", "C", "D"), (4.0, 3.0, 1.0, 0.0), depth=2)
specificity = ppm.evaluate_outcome_specificity(
    top_two, truth, (other_truth,), estimand="exact", strata=("all",) * 4
)
print(specificity.own_adjusted_recovery)  # 0.5: recovery 1 minus chance 0.5 (two slots of four)
print(specificity.mean_foreign_adjusted_recovery)  # -0.5: recovery 0 minus chance 0.5
print(specificity.specificity)  # 1.0: 0.5 - (-0.5)
```

Specificity near zero means the selection favours its own signal no more than the others.
When all signals have the same chance expectation, as here, it lies between -1 and 1. When
their targets sit in strata with different expectations it can exceed 1: own recovery 1
against chance 0.1 and other recovery 0 against chance 0.9 give 0.9 - (-0.9) = 1.8. This is
not classifier specificity, the true negative rate.

The call raises `ValueError` unless all signals have the same signal structure (`kind`), the
same features, no target feature in common and the same number of targets (groups for
`estimand="group"`, features otherwise). Groups may differ in size.

## Graph regions

A graph region is a set of features you name within a `FeatureGraph`: for example, the members
of one pathway in an interaction network. `evaluate_graph_region` measures how much of the
region lies in the top `depth` features of a ranking of the graph's nodes.

- `node_coverage` is the fraction of the region's features that are selected.
- `edge_coverage` is the fraction of the region's edges with both ends selected. Only edges
  joining two features of the region count, and `target_edge_count` is their number.

```python
import pyplasmode as ppm

network = ppm.FeatureGraph(
    nodes=("A", "B", "C", "D", "E"),
    edges=(ppm.FeatureEdge("A", "B"), ppm.FeatureEdge("C", "D")),
)
node_ranking = ("A", "B", "E", "C", "D")
region_result = ppm.evaluate_graph_region(
    node_ranking, network, region=("A", "B", "C", "D"), depths=(2,)
)
print(region_result.at_depth[0].node_coverage)  # 0.5: A and B of four features
print(region_result.at_depth[0].edge_coverage)  # 0.5: A-B of two edges
```

A region whose features share no edges has no edge coverage to measure: `edge_coverage` is
`None` and `target_edge_count` is 0. When you average over regions, leave these out of the
edge average and report how many were left out. Counting them as zero would understate
recovery.

## Ranking stability

`rank_stability` measures how similar two or more complete rankings of the same features
are, for example from refits on resampled data. It averages over every pair of rankings:

- `mean_leading_jaccard`: `|P_k intersect Q_k| / |P_k union Q_k|`, where `P_k` and `Q_k` are
  the first `k` features of the two rankings and `k` is `depth`;
- `mean_spearman`: the Spearman correlation of positions over the whole list;
- `mean_rank_biased_overlap`: rank-biased overlap (RBO), which weights agreement near the top
  more heavily.

Two random rankings share `k^2 / |U|` of their first `k` features on average, where `|U|` is
the number of features (`expected_random_intersection`). `excess_leading_intersection` is
the observed mean shared count minus this. Unlike `depths` elsewhere, `depth` is one
integer.

```python
stability = ppm.rank_stability((("A", "B", "C", "D"), ranking), depth=2)
print(round(stability.mean_leading_jaccard, 3))  # 0.333: A shared, of A, B and C
print(stability.excess_leading_intersection)  # 0.0: one shared, one expected (2^2 / 4)
print(round(stability.mean_spearman, 3))  # 0.8: B and C swap places
print(round(stability.mean_rank_biased_overlap, 3))  # 0.955
```

Extrapolated RBO is `(1-p) sum(d=1..D, p^(d-1) A_d) + p^D A_D`, where:

- `p` is `persistence`, default 0.9: each rank weighs `p` times the rank above, which at 0.9
  corresponds to reading 10 ranks on average;
- `D` is the length of the rankings;
- `A_d` is the number of features shared by the first `d` of both rankings, divided by `d`.

This is the extrapolated score in
[Webber et al., Equation 23](references.md#ranking-comparison). For rankings with ties,
use [`rank_biased_overlap`](metrics.md#consensus-and-rank-biased-overlap).

## Prediction and panels

`evaluate_prediction(target, prediction, metric=...)` scores predictions you supply with
scikit-learn and returns the score in `value`. `metric` is one of `"roc_auc"` (AUC, the area
under the ROC curve), `"average_precision"`, `"log_loss"`, `"r2"` (R-squared) or `"rmse"`
(root mean squared error). Targets and predictions must be finite, non-empty and of equal
length. A metric that is undefined for the data raises `ValueError`:

- AUC requires both binary classes;
- average precision requires a positive observation;
- R-squared requires at least two observations with a varying target;
- log loss requires binary labels and probabilities in `[0, 1]`, and accepts a single
  observed class.

`evaluate_panel` scores a reference model's predictions, for example from a model using
every feature, and a panel's predictions on the same target. The reference comes first:

```python
panel_comparison = ppm.evaluate_panel(
    [0, 0, 1, 1],  # target
    [0.1, 0.4, 0.35, 0.8],  # reference_prediction
    [0.2, 0.3, 0.6, 0.7],  # panel_prediction
    metric="roc_auc",
)
print(panel_comparison.reference_value)  # 0.75: 3 of 4 positive-negative pairs ordered
print(panel_comparison.panel_value)  # 1.0
print(panel_comparison.difference)  # 0.25: panel_value - reference_value
```

`difference` is always `panel_value - reference_value`, so swapping the two prediction
arguments flips its sign. A positive difference favours the panel for AUC, average precision
and R-squared; a negative difference favours it for log loss and root mean squared error.

## Signal reconstruction

A panel of correlates can carry the generating score without containing the generating
features. The generating score is the number computed for each row from the generating
features (`truth.signal`). `evaluate_signal_reconstruction` fits a linear regression from the
panel to the score on some rows and measures it on others. An R-squared of 0.4 means the
panel explains 40% of the score's variation in those rows.

The function takes one matrix `X` with its `feature_ids`, the panel as `selected_features`,
the score for every row of `X` as `signal`, and two disjoint sets of 0-based row indices
into `X`:

- `development_indices`: rows used to fit median imputation and ordinary least squares;
- `evaluation_indices`: rows used to compute `r_squared` and `root_mean_squared_error`.

```python
development = np.arange(70)
held_out = np.arange(70, 100)
for selected in (("A",), ("C", "D")):
    reconstruction = ppm.evaluate_signal_reconstruction(
        population.X,
        population.feature_ids,
        selected_features=selected,
        signal=truth.signal,
        development_indices=development,
        evaluation_indices=held_out,
    )
    print(
        selected,
        round(reconstruction.r_squared, 3),
        round(reconstruction.root_mean_squared_error, 3),
    )
# ('A',) 0.987 0.115: A carries weight 9 of 10
# ('C', 'D') -0.053 1.027: worse than predicting the mean
```

For generated training and test samples, stack them: `X = np.vstack([train.X, test.X])`,
`signal = np.concatenate([train.truth.signal, test.truth.signal])`, development rows
`0` to `len(train.X) - 1` and evaluation rows the rest.

An empty panel predicts the development mean. Evaluation needs at least two rows and a
score that varies across them.

## Repeated simulations

`summarize_replicates` reports the mean, Monte Carlo standard error and a 95% t interval
from at least two finite replicate estimates. For `R` independent replicates, the standard
error is `sd(estimates)/sqrt(R)` and the interval is the mean plus/minus
`t_(R-1,0.975)` times that standard error. Report this Monte Carlo uncertainty across
datasets of the same design
([Morris et al., Section 5.2](references.md#simulation-studies)). Supply the number of
failed runs through `failure_count` so they remain part of the summary.

### Paired procedure comparisons

To compare two procedures, run both on the same independently generated simulated datasets
and give each dataset an ID. `compare_paired_replicates` aligns by unique IDs before
computing right-minus-left differences.

```python
from pyplasmode import compare_paired_replicates

comparison = compare_paired_replicates(
    ("world_1", "world_2", "world_3"),
    [0.6, 0.7, 0.8],
    ("world_3", "world_1", "world_2"),
    [0.85, 0.65, 0.75],
)
print(comparison.summary.mean)  # 0.05
```

Unmatched IDs raise by default. `allow_unmatched=True` uses the
intersection and reports unmatched counts. A `None` estimate marks a failed
procedure; among matched pairs, the result counts failures in each arm and pairs with at
least one failure. The t interval is
conditional on complete successful pairs; selective failures may bias that
comparison, so report failure rates and their causes separately.
The function requires at least two complete pairs. A positive difference favours
the right procedure for AUC but favours the left procedure for a loss metric.
Refits or observations within one simulated dataset are not independent replicates;
for uncertainty over observations with fixed models, use the
[paired bootstrap](metrics.md#paired-bootstrap-and-discrimination-gap).
