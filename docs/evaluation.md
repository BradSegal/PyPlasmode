# Evaluation

Ask two things of a fitted method: how well does it predict the outcome, and did it select the
features that produced it? Prediction metrics answer the first. Recovery metrics answer the
second, which only a simulation can, because only there is the signal known: the *generating
features* whose values produced the outcome (the API calls the signal `truth`). A *panel* is the
subset of features a method selects to predict with.

Score every method you compare on the same simulated dataset, so that they face the same
outcomes and noise and a difference between their scores comes from the methods, not the draw.

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

[Metrics](metrics.md) compares selections from repeated fits, computes AUC when follow-up is
censored, and puts an interval on the difference between two fitted models.

## Feature and group recovery

*Recovery* answers one question: of what generated the outcome, how much did the method select?

**How it is used.** You pass a *ranking*, every feature once, most important first (such as the
features sorted by importance score), the signal and increasing `depths`. For each `k` in
`depths` the function:

1. treats the first `k` features as the selection;
2. counts the generating features in it (`evaluate_ranking`), the groups with a member in it
   (`evaluate_groups`) or the share of absolute weight in `truth.weights` it carries
   (`evaluate_module`);
3. returns one result per depth in `at_depth`, in the order of `depths`, so one call shows how
   recovery grows with a larger panel.

Here A and B generate the outcome, A with nine times B's weight, and a model ranks A and C first:

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

The population is 100 rows of four unrelated features; `materialize_truth` computes the signal
without simulating an outcome. The same top two, A and C, score three ways:

| Recovery | Counts | Value |
| :-- | :-- | --: |
| Exact (`recall`) | A, of two generating features | 0.5 |
| Weighted (`weighted_coverage`) | A's weight, 9 of 10 | 0.9 |
| Group, if groups are {A} and {B, C} | both groups | 1.0 |

Exact recovery counts C as a miss. Group recovery accepts it in place of B, as it should if C
is a *correlate* of B, a measured feature that predicts almost as well when strongly correlated;
a [correlated signal](truths.md#correlated) declares such groups, and the [worked recovery
script](https://github.com/BradSegal/PyPlasmode/blob/main/examples/recovery.py) scores all three.
Precision, 0.5, is the generating share of the selection.

- **When importance scores tie**, their order is arbitrary and recovery would depend on it: use
  [Rankings and scores](#rankings-and-scores).
- **A signal spread over whole groups**, such as group means or a
  [distributed signal](truths.md#distributed), has no single feature to find, so
  `evaluate_ranking` raises `ValueError` (`truth.direct_features` is empty). Use group or
  weighted recovery.

## Rankings and scores

This answers one question: how do you score a selection when several features share an
importance score, as when a tree model splits equally often on two copies of a measurement? A
ranking must put one first, and recovery would then reward or penalise an order the model never
chose.

**How it is used.** `nomination_from_scores(feature_ids, scores, depth=...)` takes non-negative
scores and the number of slots and returns the selection (a *nomination* in the API). It gives
each feature a *membership*, the probability that it is selected when tied slots are filled at
random:

1. the cutoff is the `depth`-th highest positive score;
2. a score above it has membership 1;
3. scores tied at it share the remaining slots equally;
4. lower scores have membership 0. A zero score usually means the model did not use the
   feature, so it is never selected; with fewer than `depth` positive scores the selection is
   smaller than `depth`.

Here A scores highest and B, C and D tie for the two slots left at `depth=3`:

```python
tied = ppm.nomination_from_scores(("A", "B", "C", "D"), (5.0, 2.0, 2.0, 2.0), depth=3)
print(tied.membership.round(3))
# [1.    0.667 0.667 0.667]
print(tied.selected_features, tied.achieved_size, round(tied.membership_budget, 3))
# ('A', 'B', 'C', 'D') 4 3.0
```

Four features have membership above zero (`selected_features`) for three slots
(`membership_budget`, the sum of memberships).

`evaluate_fractional_recovery(nomination, truth, estimand=...)` averages recovery over every way
of filling the tied slots and returns it in `value`. `estimand` is `"exact"`, `"group"`,
`"module"` (weighted recovery) or `"interaction"`. With the signal above (A weight 9, B 1):

```python
for estimand in ("exact", "module", "interaction"):
    recovery = ppm.evaluate_fractional_recovery(tied, truth, estimand=estimand)
    print(estimand, round(recovery.value, 3))
# exact 0.833
# module 0.967
# interaction 0.667
```

The two slots after A go to {B, C}, {B, D} or {C, D}, equally likely. The last two rows are
targets another signal could have:

| Target | How ties count | Value |
| :-- | :-- | --: |
| Exact | (1 + 2/3) / 2 | 0.833 |
| Weighted | (9 × 1 + 1 × 2/3) / 10 | 0.967 |
| Interaction of A and B | fillings with B: 2 of 3 | 0.667 |
| A group {B, C} | fillings with B or C: 3 of 3 | 1 |
| An interaction of B and C | fillings with both: 1 of 3 | 0.333 |

Exact and weighted recovery average each feature's membership. A group needs *any* member and an
interaction *both*, so they count fillings: the group scores 1 though each member has
membership 2/3, and B with C scores below the product of memberships, 4/9, because once B takes
a slot only one remains for C and D.

- **Group recovery needs a signal that declares groups, and interaction recovery one with
  exactly two generating features**; otherwise the call raises `ValueError`.
- **The selection's features must be the population's, in the same order**, so each membership
  meets the right feature. If your method scored only some, pass `allow_subset=True`: unscored
  features get membership 0 and every generating feature still counts in the denominator.

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

Outcome specificity answers one question: does the selection recover the signal behind its
outcome better than other signals on the same features?

**Why it is needed.** A method can recover generating features without using the outcome, for
example by favouring high-variance features or hubs correlated with many others. It then also
recovers signals it never saw.

**How it is used.** You pass the selection, `own_truth` (the signal behind the outcome the
method was fitted to), `foreign_truths` (a tuple of other signals), an `estimand` and strata.
`evaluate_outcome_specificity` then:

1. computes recovery above chance for each signal, as `evaluate_matched_recovery` does;
2. returns `own_adjusted_recovery`, the mean over the others
   (`mean_foreign_adjusted_recovery`) and `specificity`, the first minus the second.

Here the other signal is generated by C and D, and the method selects A and B:

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

| Signal | Recovery | Chance (any two of four) | Above chance |
| :-- | --: | --: | --: |
| Own: A and B | 1 | 0.5 | 0.5 |
| Other: C and D | 0 | 0.5 | -0.5 |

Specificity is 0.5 - (-0.5) = 1.0; near zero, the selection favours its own signal no more
than the others. With equal chance expectations, as here, it lies between -1 and 1; with
unequal ones it can exceed 1 (own recovery 1 against chance 0.1, other 0 against 0.9, give
0.9 - (-0.9) = 1.8). It is not a classifier's specificity, the true negative rate.

The call raises `ValueError` unless the signals have:

- **the same signal structure (`kind`) and features**, so one selection and one set of strata
  apply to all;
- **no feature that earns credit in two signals** (for group recovery any group member, for
  weighted recovery any weighted feature), because selecting it would count for both;
- **the same number of targets** (groups for `estimand="group"`, features otherwise), because
  finding one of two and one of ten are different achievements. Groups may differ in size,
  since the chance expectation accounts for it.

## Graph regions

A graph region answers one question: when a signal lives in a connected part of a network, did
the top features land on connected members of it, or only on scattered ones?

A `FeatureGraph` is the network: features as nodes, and pairs of related features as edges, for
example proteins that bind each other. An edge has no direction, and can carry a `weight` for
the strength of the relationship (default 1.0). A *region* is the part of the network the signal
concerns, such as one pathway: a set of nodes you name.

`evaluate_graph_region(ranking, graph, region=..., depths=...)` takes a ranking of all the
graph's nodes. For the top `depth` features it:

1. counts the region's nodes that are selected, and divides by the region's size
   (`node_coverage`);
2. finds the edges that join two nodes of the region (`target_edge_count`);
3. counts those edges with both ends selected, and divides by their number (`edge_coverage`);
4. sums the weights of those recovered edges, and divides by the summed weight of the region's
   edges (`weighted_edge_coverage`).

Two methods each put two members of a four-protein signalling chain in their top three:

```python
import pyplasmode as ppm

network = ppm.FeatureGraph(
    nodes=("EGFR", "GRB2", "SOS1", "KRAS", "TP53", "MDM2", "MYC"),
    edges=(
        ppm.FeatureEdge("EGFR", "GRB2", weight=0.9),
        ppm.FeatureEdge("GRB2", "SOS1", weight=0.3),
        ppm.FeatureEdge("SOS1", "KRAS", weight=0.3),
        ppm.FeatureEdge("TP53", "MDM2", weight=0.8),
    ),
)
pathway = ("EGFR", "GRB2", "SOS1", "KRAS")
method_a = ("EGFR", "GRB2", "TP53", "SOS1", "KRAS", "MDM2", "MYC")
method_b = ("EGFR", "SOS1", "TP53", "GRB2", "KRAS", "MDM2", "MYC")

region_result = ppm.evaluate_graph_region(method_a, network, region=pathway, depths=(3,))
other_result = ppm.evaluate_graph_region(method_b, network, region=pathway, depths=(3,))
for row in (region_result.at_depth[0], other_result.at_depth[0]):
    print(row.node_coverage, round(row.edge_coverage, 2), round(row.weighted_edge_coverage, 2))
# 0.5 0.33 0.6
# 0.5 0.0 0.0
```

| Top three | Node coverage | Edge coverage | Weighted edge coverage |
| --- | --- | --- | --- |
| Method A: EGFR, GRB2, TP53 | 0.50 (2 of 4) | 0.33 (1 of 3 edges) | 0.60 (0.9 of 1.5) |
| Method B: EGFR, SOS1, TP53 | 0.50 (2 of 4) | 0.00 | 0.00 |

Both methods found half the pathway. Method A found an interacting pair, and the strongest
one. Method B found two members that do not touch. Node coverage cannot tell them apart; edge
coverage can. The TP53 to MDM2 edge is outside the region, so it never counts.

Two rules:

- **A region with no internal edges has no edge coverage.** Both edge figures are `None` and
  `target_edge_count` is 0. When you average over regions, leave these out and report how many
  were left out. Counting them as zero would understate recovery.
- **Use the weighted figure only when the weights measure what matters to the signal.** If they
  are confidence scores from an annotation database, read `edge_coverage`, or drop
  low-confidence edges before building the graph.

## Ranking stability

Ranking stability answers one question: if you refit the method, on bootstrap samples or
cross-validation folds, does it rank the features the same way? A ranking that changes with
each refit cannot be relied on to select the same features on new data, even if one fit
recovered well.

**How it is used.** You pass two or more complete rankings of the same features, without ties,
and one integer `depth`. `rank_stability` compares every pair and averages the measures in the
table below. Here the ranking from the first section is compared with the alphabetical one:

```python
stability = ppm.rank_stability((("A", "B", "C", "D"), ranking), depth=2)
print(round(stability.mean_leading_jaccard, 3), stability.excess_leading_intersection)
# 0.333 0.0
print(round(stability.mean_spearman, 3), round(stability.mean_rank_biased_overlap, 3))
# 0.8 0.955
```

The two differ only by B and C swapping places across the cutoff:

| Measure | What it compares | Value |
| :-- | :-- | --: |
| `mean_leading_jaccard` | share of features in either top `depth` that are in both: A of A, B, C | 0.333 |
| `excess_leading_intersection` | features shared in the top `depth` minus the `depth^2 / n` random rankings of `n` features share: 1 - 2^2 / 4 | 0.0 |
| `mean_spearman` | Spearman correlation of positions: 1 same order, -1 reversed | 0.8 |
| `mean_rank_biased_overlap` | rank-biased overlap (RBO): the whole ranking, the top weighted most | 0.955 |

The leading measures see a large disagreement, no better than random, because the swap crosses
the cutoff; the whole-ranking measures see one small swap. In RBO each rank counts
`persistence` (default 0.9) times as much as the rank above; lower it to judge only the top
few. RBO here is the extrapolated score of
[Webber et al., Equation 23](references.md#ranking-comparison). For tied rankings use
[`rank_biased_overlap`](metrics.md#consensus-and-rank-biased-overlap).

## Prediction and panels

Recovery shows which features a method found; prediction metrics show whether the model built
on them is useful. `evaluate_prediction(target, prediction, metric=...)` scores your predictions
with scikit-learn and returns `value`. `metric` is `"roc_auc"` (AUC, the area under the ROC
curve), `"average_precision"` or `"log_loss"` for a binary outcome, and `"r2"` (R-squared) or
`"rmse"` (root mean squared error) for a continuous one. An undefined metric raises `ValueError`
rather than return a misleading number: AUC compares positive with negative observations, so
needs both classes; average precision needs a positive; R-squared needs two different targets.

A panel is worth having if it predicts nearly as well as a model on every feature.
`evaluate_panel` scores a reference model's predictions and a panel's on one target, reference
first:

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

The reference ranks three of the four positive-negative pairs correctly (the positive at 0.35
falls below the negative at 0.4); the panel ranks all four. `difference` is always
`panel_value - reference_value`, so swapping the predictions flips its sign. A positive
difference favours the panel for AUC, average precision and R-squared, and a negative one for
log loss and root mean squared error, where lower is better.

## Signal reconstruction

Signal reconstruction answers one question: can the panel reproduce the *generating score*, the
number computed for each row from the generating features (`truth.signal`)? A panel of
correlates can carry that score without containing the generating features; recovery gives it
no credit, though it may predict as well as they would.

**How it is used.** You pass the matrix, its feature IDs, the panel and the generating score in
one row order, with `development_indices` and `evaluation_indices`.
`evaluate_signal_reconstruction` then:

1. fills missing values with medians from the development rows;
2. fits a linear regression from the panel to the score on those rows;
3. returns R-squared and root mean squared error on the evaluation rows only, so that a large
   panel cannot score well by fitting noise.

An R-squared of 0.4 means the panel explains 40% of the score's variation on those rows. Here 70
rows fit and 30 measure:

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

A carries 9 of the 10 units of weight, so it reconstructs almost all of the score. C and D are
unrelated: R-squared below zero means they predict the held-out score worse than the held-out
rows' own mean would.

- **An empty panel predicts the development mean**, scoring an R-squared at or slightly below zero:
  a baseline.
- **The evaluation rows must be at least two, with a score that varies**, because R-squared
  measures explained variation.
- **For samples from one `generate_partitioned` call**, whose scores share one scale, stack
  them: `X = np.vstack([train.X, test.X])` and
  `signal = np.concatenate([train.truth.signal, test.truth.signal])`. Development rows are then
  `0` to `len(train.X) - 1`, evaluation rows the rest.

## Repeated simulations

One simulated dataset gives one estimate; another drawn from the same *design* (population,
signal, outcome model and sample size) would give a different one. To know how much a result
depends on the draw, run the design several times and summarise.

`summarize_replicates` takes one estimate per successful simulation, at least two, and returns
their mean, its Monte Carlo standard error (the standard deviation of the estimates over the
square root of their number) and a 95% t interval. Pass failed simulations as `failure_count`:
the summary uses only successes and records the count, so a reader sees how many are missing.
Here four simulations gave an AUC and one failed:

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

The interval describes uncertainty about the design's average result, so it narrows with more
simulations. Report it so a reader can tell whether a difference between methods exceeds the
simulation noise
([Morris et al., Section 5.2](references.md#simulation-studies)).

### Paired procedure comparisons

This answers one question: across simulated datasets, which of two procedures does better?
Run both on the same datasets and take the difference on each. A dataset easy for one is
usually easy for the other, so the differences vary less than the separate results and fewer
simulations tell the procedures apart.

**How it is used.** You pass, for each procedure, dataset IDs and one estimate per dataset.
`compare_paired_replicates` then:

1. matches the estimates by ID, so the lists may come in different orders;
2. takes the right-minus-left difference on each dataset;
3. summarises the differences with `summarize_replicates`.

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

The right procedure wins on all three datasets, by 0.04 on average, but with three the interval
still includes zero. A positive difference favours the right procedure for AUC and the left for
a loss, where lower is better.

- **Unmatched IDs raise `ValueError`**, because a dataset scored by one procedure only usually
  means a bookkeeping error. `allow_unmatched=True` compares on the shared datasets and reports
  the unmatched counts.
- **Mark a failed run with `None`** rather than leaving it out, so it is counted. The summary
  uses only pairs where both succeeded; if failures fall on hard datasets the difference
  describes only the easier ones, so report failure counts and causes beside it.
- **At least two complete pairs are needed**, since a standard error needs two differences.
- **Each dataset must be generated independently.** Refits or observations within one dataset
  share its draw. For uncertainty over one dataset's observations with fixed models, use the
  [paired bootstrap](metrics.md#paired-bootstrap-and-discrimination-gap).
