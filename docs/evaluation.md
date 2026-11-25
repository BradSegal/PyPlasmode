# Evaluation

A simulation lets you ask two things of a fitted method: how well it predicts the outcome, and
whether it selected the features that produced the outcome. Prediction metrics answer the
first. Recovery metrics answer the second, which only a simulation can, because only there is
the signal known: the *generating features* whose values produced the outcome. The API calls
the signal `truth`. A *panel* is the subset of
features a method selects to predict with.

Score every method you compare on the same simulated dataset. The methods then face the same
outcomes and the same noise, so a difference between their scores comes from the methods and
not from the luck of the draw.

| Question | Function |
| --- | --- |
| Which generating features were found? | `evaluate_ranking` |
| Was at least one member of each relevant group found? | `evaluate_groups` |
| How much generating weight does the panel cover? | `evaluate_module` |
| How is recovery shared when importance scores tie? | `evaluate_fractional_recovery` |
| How far is recovery above chance? | `evaluate_matched_recovery` |
| Does a selection prefer its own signal to other signals? | `evaluate_outcome_specificity` |
| What fraction of a graph region's nodes and edges was recovered? | `evaluate_graph_region` |
| How similar are repeated rankings? | `rank_stability` |
| How accurate are the predictions? | `evaluate_prediction` |
| How does a smaller panel compare with a reference model? | `evaluate_panel` |
| Can the panel reconstruct the generating score? | `evaluate_signal_reconstruction` |
| What is the average result across simulations? | `summarize_replicates` |
| Which of two procedures does better across simulations? | `compare_paired_replicates` |

[Metrics](metrics.md) has the measures that compare selections from repeated fits, compute AUC
when follow-up is censored, and put an interval on the difference between two fitted models.

## Feature and group recovery

*Recovery* answers whether a method selected the features that generated the outcome: it is the
fraction of the generating features, of their groups or of their weight that the method's
selection contains.

The recovery functions take a *ranking*: a tuple of every feature once, most important first, such
as the features sorted by a model's importance scores. For each `k` in `depths`, given in increasing
order, they treat the first `k` features as the selection and score it, so one call shows how
recovery grows as you allow a larger panel. If several features have the same importance score,
their order in the ranking is arbitrary and the recovery would depend on it. For tied scores, use
[Rankings and scores](#rankings-and-scores) instead.

In this example A and B generate the outcome, A with nine times B's weight, and a model ranks
A and C first:

```python
import numpy as np
import pyplasmode as ppm

population = ppm.Population(np.random.default_rng(7).normal(size=(100, 4)), ("A", "B", "C", "D"))
truth = ppm.materialize_truth(
    ppm.SparseTruth(features=("A", "B"), weights=(9.0, 1.0)), population, seed=3
)
ranking = ("A", "C", "B", "D")
result = ppm.evaluate_ranking(ranking, truth, depths=(2,))
print(result.at_depth[0].recall, result.at_depth[0].precision)
# 0.5 0.5
module = ppm.evaluate_module(ranking, truth, depths=(2,))
print(module.at_depth[0].weighted_coverage)
# 0.9
```

The population is 100 rows of four unrelated random features. `materialize_truth` computes the
signal from its specification without simulating an outcome; its `seed` matters only for
specifications that choose their own features or custom signals that draw random numbers.
`evaluate_ranking` scores the top two features, A and C. A is one of the two generating features, so
recovery is 0.5 (the API calls it `recall`). One of the two selected features is a generating
feature, so precision is also 0.5. `evaluate_module` counts weight instead of features: A carries 9
of the total weight of 10, so weighted recovery is 0.9. `at_depth` holds one result per depth, in
the order of `depths`.

A model can also select a *correlate* of a generating feature: a measured feature correlated with
it, which predicts almost as well when the correlation is strong. If C is a correlate of B,
selecting C instead of B may be acceptable. A [correlated signal](truths.md#correlated) can declare
groups such as {A} and {B, C}, and group recovery then counts a group as found when any member is
selected: with A and C selected, both groups are found. The [worked recovery
script](https://github.com/BradSegal/PyPlasmode/blob/main/examples/recovery.py) scores exact, group
and weighted recovery on this population.

A signal spread over whole groups, such as group means or a
[distributed signal](truths.md#distributed), has no single feature to find, so
`truth.direct_features` is empty and `evaluate_ranking` raises `ValueError`. Measure such a
signal with group recovery, or with weighted recovery, which divides the absolute weight of the
selected features by the total absolute weight in `truth.weights`.

## Rankings and scores

Models often give several features the same importance score, for example when a tree model splits
equally often on two copies of a measurement. A ranking must put one of them first, and recovery
would then reward or penalise an order the model never chose. Instead,
`nomination_from_scores(feature_ids, scores, depth=...)` shares the contested slots equally among
the tied features. It takes non-negative importance scores and the number of slots, `depth`, and
returns the selection (the API calls a selection a *nomination*). Each feature gets a *membership*:
the probability that it is selected when the tied slots are filled at random. The cutoff is the
`depth`-th highest positive score.

- A score above the cutoff has membership 1.
- Scores tied at the cutoff share the remaining slots equally.
- Lower scores have membership 0. A zero score usually means the model did not use the feature,
  so it is never selected, and with fewer than `depth` positive scores the selection is smaller
  than `depth`.

Here A scores highest and B, C and D tie for the two slots left at `depth=3`:

```python
tied = ppm.nomination_from_scores(("A", "B", "C", "D"), (5.0, 2.0, 2.0, 2.0), depth=3)
print(tied.membership.round(3))
# [1.    0.667 0.667 0.667]
print(tied.selected_features, tied.achieved_size, round(tied.membership_budget, 3))
# ('A', 'B', 'C', 'D') 4 3.0
```

`selected_features` lists every feature with membership above zero and `achieved_size` counts
them, here four features for three slots. `membership_budget` sums the memberships, which is
the number of slots filled.

`evaluate_fractional_recovery(nomination, truth, estimand=...)` averages recovery over every way of
filling the tied slots and returns it in `value`. `estimand` names the recovery measure: `"exact"`
(recovery of the generating features), `"group"`, `"module"` (weighted recovery) or `"interaction"`.
With the signal from the previous section, in which A has weight 9 and B weight 1:

```python
for estimand in ("exact", "module", "interaction"):
    recovery = ppm.evaluate_fractional_recovery(tied, truth, estimand=estimand)
    print(estimand, round(recovery.value, 3))
# exact 0.833
# module 0.967
# interaction 0.667
```

Recovery and weighted recovery are averages over the generating features, so each feature
counts its membership instead of 0 or 1. A counts 1 and B counts 2/3, so recovery is
(1 + 2/3) / 2 and weighted recovery is (9 × 1 + 1 × 2/3) / 10.

Group and interaction recovery cannot be averaged this way, because a group is found when *any*
member is selected and an interaction only when *both* features are. They count the equally
likely ways to fill the tied slots instead. In the example the two slots after A go to {B, C},
{B, D} or {C, D}:

- The interaction of A and B is recovered whenever B is chosen: in two of the three fillings, so
  2/3.
- A group {B, C} would be found in every filling, so 1, although each member has membership 2/3.
- An interaction of B and C would be recovered only by {B, C}: 1/3. That is less than the product
  of the memberships, 4/9, because once B takes a slot only one slot is left for C and D.

Group recovery needs a signal that declares groups, and interaction recovery a signal with
exactly two generating features; otherwise the call raises `ValueError`. The selection's
features must be the population's features in the same order, so that each membership meets
the right feature. If your method scored only some of them, for example after filtering, pass
`allow_subset=True`: the unscored features get membership 0, and every generating feature still
counts in the denominator.

## Matched chance recovery

Some recovery happens by luck. Selecting ten of 100 features at random recovers 10% of the
generating features on average, so a method that recovers 40% has done 30 percentage points
better than chance, not 40. *Recovery above chance* is the recovery observed minus the recovery
a random selection would get. The API calls it `chance_adjusted_recovery`.

The answer depends on what "random" means. If one kind of feature is more likely to be a
generating feature, or easier to hit, a selection that favours that kind scores well without
having identified anything. A *stratum* is a label you give each feature to say which features
the random selection may swap for which, so that it resembles the real selection in kind.

`evaluate_matched_recovery(nomination, truth, estimand=..., strata=...)` takes a selection from
`nomination_from_scores`, an `estimand` as in `evaluate_fractional_recovery`, and one stratum
label per feature, in the order of the selection's features. It then:

1. counts the selections in each stratum;
2. builds the random reference by keeping those counts and reshuffling which features within
   each stratum are selected;
3. computes the expected recovery of that reference exactly, without simulation;
4. returns `observed_recovery`, `expected_recovery` and their difference,
   `chance_adjusted_recovery`. Zero means no better than chance and a negative value worse.

The tutorial works through an example with its numbers in
[Compare with matched chance](tutorial.md#compare-with-matched-chance), and
[Matching strata on empirical measurements](tutorial.md#matching-strata-on-empirical-measurements)
derives strata from measured features.

Three rules for choosing strata:

- **Use the measurements only**, never the outcome, the signal or a fitted model, or the
  reference can be tuned to flatter a method.
- **Fix them before looking at results.**
- **Keep them coarse enough that chance can vary.** `admissible_panel_count` is the number of
  ways the reference can rearrange the selection within its strata. At 1 the only
  rearrangement is the observed selection, so recovery above chance is zero whatever the
  method did.

When scores tie at the cutoff, the reference keeps the shared memberships as well as the
counts. For group recovery the expectation accounts for group size, because a random selection
is more likely to hit a large group than a small one.

### Outcome specificity

A method can recover generating features without using the outcome, for example by favouring
high-variance features or hubs correlated with many others. Such a method also recovers signals
it never saw. To test for this, take other signals on the same features and score the one
selection against all of them: a selection driven by the outcome should recover its own signal
better than the others.

`evaluate_outcome_specificity` takes the selection, `own_truth` (the signal behind the outcome
the method was fitted to) and `foreign_truths` (a tuple of other signals). It computes recovery
above chance for each signal as `evaluate_matched_recovery` does, and returns
`own_adjusted_recovery`, the mean for the other signals (`mean_foreign_adjusted_recovery`) and
`specificity`, the first minus the second. Here the other signal is generated by C and D, and
the method selects A and B:

```python
other_truth = ppm.materialize_truth(ppm.SparseTruth(features=("C", "D")), population, seed=3)
top_two = ppm.nomination_from_scores(("A", "B", "C", "D"), (4.0, 3.0, 1.0, 0.0), depth=2)
specificity = ppm.evaluate_outcome_specificity(
    top_two, truth, (other_truth,), estimand="exact", strata=("all",) * 4
)
print(
    specificity.own_adjusted_recovery,
    specificity.mean_foreign_adjusted_recovery,
    specificity.specificity,
)
# 0.5 -0.5 1.0
```

All four features share one stratum, so the random reference is two features drawn from four,
which recovers half of any two-feature signal. The selection recovers all of its own signal,
0.5 above chance, and none of the other, 0.5 below, so specificity is 1.0.

Specificity near zero means the selection favours its own signal no more than the others. When
every signal has the same chance expectation, as here, specificity lies between -1 and 1. It
can exceed 1 when the signals' generating features sit in strata with different expectations:
own recovery 1 against chance 0.1 and other recovery 0 against chance 0.9 give
0.9 - (-0.9) = 1.8. The name does not refer to a classifier's specificity, the true negative
rate.

The signals must be comparable, so the call raises `ValueError` unless:

- they have the same signal structure (`kind`) and the same features, so that one selection and
  one set of strata apply to all of them;
- no feature that earns recovery credit belongs to two signals, because selecting it would count
  for both. For group recovery this covers every group member, and for weighted recovery every
  weighted feature;
- they have the same number of targets (groups for `estimand="group"`, features otherwise),
  because finding one of two targets and one of ten are different achievements. Groups may
  differ in size, since the chance expectation accounts for it.

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

A method whose ranking changes each time you refit it, on bootstrap samples or cross-validation
folds, cannot be relied on to select the same features on new data, even if one fit recovered
well. `rank_stability` compares every pair of two or more complete rankings of the same
features and averages three measures:

- `mean_leading_jaccard`: of the features in the first `depth` of either ranking, the share that
  are in the first `depth` of both. It ignores the order within those features.
- `mean_spearman`: the Spearman correlation of the features' positions over the whole ranking:
  1 for the same order, -1 for the reverse.
- `mean_rank_biased_overlap`: rank-biased overlap (RBO), which compares the whole ranking but
  counts agreement near the top more heavily.

Some leading features are shared by chance. Two random rankings of `n` features share
`depth^2 / n` of their first `depth` features on average (`expected_random_intersection`), so
`excess_leading_intersection`, the observed mean number shared minus this, is near zero when the
rankings agree no more than random ones. Unlike `depths` in the recovery functions, `depth` is a
single integer.

Comparing the ranking from the first section with the alphabetical one:

```python
stability = ppm.rank_stability((("A", "B", "C", "D"), ranking), depth=2)
print(round(stability.mean_leading_jaccard, 3), stability.excess_leading_intersection)
# 0.333 0.0
print(round(stability.mean_spearman, 3), round(stability.mean_rank_biased_overlap, 3))
# 0.8 0.955
```

The first two features are A, B in one ranking and A, C in the other: one shared feature of the
three that appear, so the Jaccard overlap is 1/3. Two random rankings of four features share
2^2 / 4 = 1 of their first two on average, as these do, so the excess is zero. Swapping B and
C lowers the Spearman correlation to 0.8.

In RBO each rank counts `persistence` (default 0.9) times as much as the rank above it; lower
it to judge agreement over the top few features only. RBO here is the extrapolated score in
[Webber et al., Equation 23](references.md#ranking-comparison). The rankings cannot contain
ties; for tied rankings use [`rank_biased_overlap`](metrics.md#consensus-and-rank-biased-overlap).

## Prediction and panels

Recovery shows which features a method found; prediction metrics show whether the model built on
them is useful. `evaluate_prediction(target, prediction, metric=...)` scores predictions you
supply, using scikit-learn, and returns the score in `value`. `metric` is one of:

- for a binary outcome, `"roc_auc"` (AUC, the area under the ROC curve), `"average_precision"`
  or `"log_loss"`;
- for a continuous outcome, `"r2"` (R-squared) or `"rmse"` (root mean squared error).

Where the metric is undefined, the call raises `ValueError` rather than return a misleading
number: AUC compares positive with negative observations, so it needs both classes; average
precision needs at least one positive observation; and R-squared needs at least two different
targets.

A panel is worth having if it predicts nearly as well as a model that uses every feature.
`evaluate_panel` scores a reference model's predictions and a panel's predictions on the same
target. The reference comes first:

```python
panel_comparison = ppm.evaluate_panel(
    [0, 0, 1, 1],  # target
    [0.1, 0.4, 0.35, 0.8],  # reference_prediction
    [0.2, 0.3, 0.6, 0.7],  # panel_prediction
    metric="roc_auc",
)
print(panel_comparison.reference_value, panel_comparison.panel_value, panel_comparison.difference)
# 0.75 1.0 0.25
```

The reference ranks three of the four positive-negative pairs correctly: the positive at 0.35
falls below the negative at 0.4. The panel ranks all four correctly. `difference` is always
`panel_value - reference_value`, so swapping the two prediction arguments flips its sign. A
positive difference favours the panel for AUC, average precision and R-squared, where higher is
better, and a negative difference favours it for log loss and root mean squared error, where
lower is better.

## Signal reconstruction

A panel of correlates can carry the generating score without containing the generating features. The
*generating score* is the number computed for each row from the generating features
(`truth.signal`). Recovery of the generating features gives such a panel no credit, though it may
predict as well as they would. `evaluate_signal_reconstruction` measures this directly: it fits a
linear regression from the panel to the generating score on some rows and measures the fit on
others. An R-squared of 0.4 means the panel explains 40% of the score's variation on those rows.

Pass the matrix, its feature IDs and the generating score in the same row order. The function
fits on the rows in `development_indices`, filling missing values with medians from those rows,
and scores only the rows in `evaluation_indices`. Fitting and measuring on different rows keeps
a large panel from scoring well by fitting noise. Here the first 70 rows are for fitting and the
last 30 for measuring, for A and for the unrelated C and D:

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
# ('A',) 0.987 0.115
# ('C', 'D') -0.053 1.027
```

A carries 9 of the 10 units of generating weight, so A alone reconstructs almost all of the
score. C and D are unrelated to it: their R-squared is below zero, which means they predict the
held-out score worse than a constant at the held-out rows' own mean would.

For training and test samples from one `generate_partitioned` call, whose generating scores
share one scale, stack them into one matrix:
`X = np.vstack([train.X, test.X])` and
`signal = np.concatenate([train.truth.signal, test.truth.signal])`. The development rows are
then `0` to `len(train.X) - 1` and the evaluation rows the rest.

An empty panel predicts the development mean, so it scores an R-squared at or just below zero and
serves as a baseline. R-squared measures explained variation, so the evaluation rows must number at
least two and have a generating score that varies across them.

## Repeated simulations

One simulated dataset gives one estimate, and another dataset drawn from the same *design* (the
same population, signal, outcome model and sample size) would give a different one. To know how
much a result depends on the draw, run the design several times and summarise the estimates.

`summarize_replicates` takes one estimate per successful simulation, at least two, and returns
their mean, its Monte Carlo standard error and a 95% t interval. Pass the number of failed
simulations as `failure_count`. The summary uses only the successful ones, and records the count
so that a reader can see how many are missing. Here four simulations gave an AUC and one failed:

```python
replicate_summary = ppm.summarize_replicates([0.71, 0.68, 0.74, 0.70], failure_count=1)
print(round(replicate_summary.mean, 3), round(replicate_summary.standard_error, 3))
# 0.708 0.012
print(
    round(replicate_summary.confidence_interval_low, 3),
    round(replicate_summary.confidence_interval_high, 3),
)
# 0.668 0.747
```

The standard error is the standard deviation of the estimates divided by the square root of
their number. The interval describes uncertainty about the design's average result, so it
narrows as you run more simulations. Report it with your results so a reader can tell whether a
difference between methods is larger than the simulation noise
([Morris et al., Section 5.2](references.md#simulation-studies)).

### Paired procedure comparisons

To compare two procedures, run both on the same simulated datasets and take the difference on
each. A dataset that is easy for one procedure is usually easy for the other, so the
differences vary less than the procedures' separate results, and fewer simulations tell them
apart.

`compare_paired_replicates` takes, for each procedure, the dataset IDs and one estimate per
dataset. It matches the estimates by ID, so the two lists may come in different orders, and
summarises the right-minus-left differences with `summarize_replicates`:

```python
comparison = ppm.compare_paired_replicates(
    ("world_1", "world_2", "world_3"),
    [0.60, 0.70, 0.80],
    ("world_3", "world_1", "world_2"),
    [0.84, 0.66, 0.72],
)
print([round(value, 3) for value in comparison.differences])
# [0.06, 0.02, 0.04]
print(
    round(comparison.summary.mean, 3),
    round(comparison.summary.confidence_interval_low, 3),
    round(comparison.summary.confidence_interval_high, 3),
)
# 0.04 -0.01 0.09
```

The right procedure does better on all three datasets, by 0.04 on average, but with three
datasets the interval still includes zero. A positive difference favours the right procedure for
AUC and the left procedure for a loss, where lower is better.

- **Unmatched IDs raise `ValueError`**, because a dataset scored by only one procedure usually
  means a bookkeeping error. To compare on the shared datasets anyway, pass
  `allow_unmatched=True`; the result reports how many IDs each side had unmatched.
- **Mark a failed run with `None`** instead of leaving it out. The result counts failures in each
  procedure and the pairs with at least one failure, and the summary uses only pairs in which
  both succeeded. If failures fall on hard datasets, the difference then describes only the
  easier ones, so report the failure counts and their causes beside it.
- **At least two complete pairs are needed**, since a standard error needs two differences.
- **Each dataset must be generated independently.** Refits or observations within one dataset
  share its draw and are not independent replicates. For uncertainty over the observations of
  one dataset with fixed models, use the
  [paired bootstrap](metrics.md#paired-bootstrap-and-discrimination-gap).
