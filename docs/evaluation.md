# Evaluation

Choose an evaluator for the question you want to answer. Prediction metrics describe the
model's accuracy; recovery metrics compare its nominations with the generating signal.

| Question | Function |
| --- | --- |
| Which generating features were found? | `evaluate_ranking` |
| Were their correlated substitutes found? | `evaluate_groups` |
| How much generating weight does the panel cover? | `evaluate_module` |
| Which region nodes and edges were recovered? | `evaluate_graph_region` |
| Can the panel reconstruct the signal? | `evaluate_signal_reconstruction` |
| How accurate are the predictions? | `evaluate_prediction` |
| How does a smaller panel compare with a reference model? | `evaluate_panel` |
| How similar are repeated rankings? | `rank_stability` |
| What is the average result across simulations? | `summarize_replicates` |

## Rankings and scores

Ranking evaluators accept a complete ordered tuple containing every feature once. Resolve
score ties using your chosen ordering rule before supplying this tuple.

Alternatively, pass non-negative importance scores to `nomination_from_scores`. It selects
the leading positive scores and shares any remaining slots equally among features tied at
the cutoff. For example, two features tied for one slot each receive membership 0.5.
Zero scores are excluded, so a model with fewer positive scores than the requested depth
returns a smaller selection.

`selected_features` contains features with non-zero membership. `positive_support` contains
all features with a positive input score, including those below the cutoff.
Use `evaluate_fractional_recovery` to compute exact, group, weighted-module or interaction
recovery from these memberships.

## Feature and group recovery

Suppose A and B generate the outcome, but a model's top two features are A and C.
It has found one of two contributors: recall is 0.5. One of its two selections is correct:
precision is also 0.5, and the false-discovery fraction is 0.5. These quantities have
different denominators even when their values happen to agree.

```python
import numpy as np
import pyplasmode as ppm

population = ppm.Population(np.random.default_rng(7).normal(size=(100, 4)), ("A", "B", "C", "D"))
truth = ppm.materialize_truth(
    ppm.SparseTruth(features=("A", "B"), weights=(9.0, 1.0)), population, seed=3
)
ranking = ("A", "C", "B", "D")
result = ppm.evaluate_ranking(ranking, truth, depths=(2,))
print(result.at_depth[0].recall)  # 0.5
print(result.at_depth[0].precision)  # 0.5
```

Now suppose A carries nine times B's generating weight. The same selection recovers half
the contributors but 90% of their absolute weight. If C is an admissible substitute for B,
the selection also recovers both groups. Exact recovery counts contributors, weighted
recovery accounts for their effect sizes, and group recovery credits the specified substitutes.

| Question | Example | Score |
| --- | --- | --- |
| Did we find the exact contributors? | A and B generate risk; A and C are selected. | Recall 1/2 |
| Are the selections contributors? | One of two selected features generates risk. | Precision 1/2 |
| Did we find their groups? | Groups are {A} and {B, C}; A and C are selected. | Group recovery 2/2 |
| How much generating weight is covered? | A has weight 9 and B weight 1; only A is selected. | Weighted coverage 9/10 |
| Was an interaction recovered? | The signal requires A and B; only A is selected. | Joint recovery 0 |

For a tied boundary, think of selecting uniformly among the tied features. If B and C tie
for one place, each has inclusion probability 0.5. If both belong to the same group, that
group is certain to be found; if they belong to different groups, each group is found with
probability 0.5. The equations below implement these probabilities without breaking ties
arbitrarily. The [worked recovery script](https://github.com/BradSegal/PyPlasmode/blob/main/examples/recovery.py)
reproduces the exact, weighted and group examples.

Let `S_k` be a nominated set of size `k`, `T` the exact generating support, and `U` the
feature universe. Exact recall is `|S_k intersect T| / |T|`; precision is
`|S_k intersect T| / k`; false-discovery fraction is `1 - precision`. All three use the
specified simulation truth. Exact recall requires a non-empty set of exact generating
features; use group or weighted recovery for distributed signals.

For fractional nomination membership `m_j`, exact recall is `sum(j in T, m_j) / |T|`.
For generating weights `w_j`, weighted coverage is
`sum(j, |w_j| m_j) / sum(j, |w_j|)`. A two-protein truth with weights `(9, 1)` therefore
has recall 0.5 and weighted coverage 0.9 when only its first protein is nominated.

For group `G`, group recovery is the probability of selecting at least one member.
At a tied boundary with `b` proteins and `r` remaining slots, if `t` group members are tied
and no member is already selected, recovery is `1 - choose(b-t, r) / choose(b, r)`.
Interaction recovery instead requires both generating proteins. If both are tied it is
`choose(b-2, r-2) / choose(b, r)`, zero when fewer than two slots remain.

`achieved_size` counts all identities with non-zero membership, including every boundary-tied
identity. It can exceed the requested depth. `membership_budget` is the sum of membership
probabilities and cannot exceed that depth.

## Matched chance recovery

Selecting ten out of 100 exchangeable features recovers 10% of the generating features
on average, even if the ranking contains no information. Recovering 40% therefore gives
an excess of 30 percentage points above that reference. Grouped features need more care:
a large group is easier to hit randomly than a small one. A matched reference keeps such
properties comparable, so improvement does not merely reflect an easier selection task.

`evaluate_matched_recovery` compares observed recovery with its expectation under random
feature assignment. You supply strata: groups of features with similar properties, defined
without using the generating truth or evaluation outcomes. The reference retains the number
of positive scores in each stratum and the selection's membership weights, while reassigning
them to feature identities within that stratum.

For example, a proteomic panel may contain strongly correlated pairs and isolated proteins.
Label each member of a pair `pair` and each isolated protein `singleton`, using the
measurements before generating outcomes. Matching within these strata compares a method
with random lists containing the same number of paired and isolated measurements.
The [biomarker tutorial](tutorial.md#compare-with-matched-chance) gives executable code
and interprets the resulting observed, expected and excess recovery.

The returned chance-adjusted recovery is `observed - expected`. For exact recall, expected
membership of a feature in stratum `h` is `sum(j in h, m_j) / |h|`. Expectations are computed
analytically for exact, group, module and interaction recovery. If only one assignment is
possible, observed and expected recovery coincide and the adjusted value is zero.

### Outcome specificity

`evaluate_outcome_specificity` scores the same nomination against its own generating truth
and other, foreign truths. This measures whether the selection recovers the particular
outcome signal more strongly than unrelated generating signals.

Own-versus-foreign specificity requires disjoint supports and equal numbers of recovery targets
under the same mechanism: groups for group recovery, features otherwise. Group membership
sizes may differ; the matched-reference expectation accounts for those sizes.
Specificity is own chance-adjusted recovery minus mean foreign chance-adjusted recovery.
As a difference between adjusted scores, it can exceed one in magnitude.

## Ranking stability

Leading-set Jaccard is `|A_k intersect B_k| / |A_k union B_k|`. The expected random
intersection is `k^2 / |U|`. Spearman correlation uses the complete rankings and requires
at least two features. Extrapolated rank-biased overlap is
`(1-p) sum(d=1..D, p^(d-1) A_d) + p^D A_D`, where `A_d` is prefix overlap divided by `d`.
The persistence parameter sets how strongly leading ranks are weighted. Its default `p=0.9`
gives geometric expected inspection depth 10. See [Webber et al.](references.md).

## Prediction and panels

`evaluate_prediction` computes ROC AUC, average precision, log loss, R-squared or root mean
squared error using scikit-learn. Supply finite, nonempty targets and predictions. ROC AUC
requires both binary classes, average precision requires a positive observation, and
R-squared requires at least two observations with a varying target. Log loss uses binary
labels and probabilities in `[0, 1]`, including when only one class is observed. Undefined
metrics raise `ValueError`.

`evaluate_panel` compares panel and reference predictions on the same target. It returns
both metric values and `difference = panel - reference`. Positive differences favour the
panel for AUC, average precision and R-squared; negative differences favour it for log loss
and root mean squared error.

## Signal reconstruction

A selected substitute may predict the generating score well even when the exact contributor
is absent. Reconstruction tests that possibility by learning a linear mapping from the panel
to the known score. An out-of-sample R-squared of 0.4 means the panel explains 40% of its
variation in the test sample. It does not mean that 40% of the generating proteins were found.

`evaluate_signal_reconstruction` fits median imputation and ordinary least squares on
development rows, then measures linear reconstruction of the known signal on evaluation
rows. An empty panel uses an intercept-only predictor. The result includes R-squared and
root mean squared error; evaluation requires at least two observations and a varying signal.

## Repeated simulations

`summarize_replicates` reports the mean, Monte Carlo standard error and a 95% t interval
from at least two finite replicate estimates. For `R` independent replicates, the standard
error is `sd(estimates)/sqrt(R)` and the interval is the mean plus/minus
`t_(R-1,0.975)` times that standard error. These describe precision across simulations of
the same design. Supply the number of failed runs through `failure_count` so they remain
part of the summary.
