# Generating signals

What counts as recovering the signal depends on how the outcome was generated: a method may be
expected to find one feature, any member of a correlated group, or most of a pathway. The
*signal* (`truth=` in code) states which features generate the outcome and in what structure.
From it PyPlasmode computes the *generating score* (`truth.signal`), one number per row, and the
recovery targets that a selection is scored against; the [outcome model](outcomes.md) sets how
strongly the score acts on the outcome.

![Five signal structures and their corresponding recovery targets](assets/concepts/signal-recovery.png)

In the diagram, "drivers" are generating features and "latent disease state" is the generating
score. The [tutorial](tutorial.md) builds sparse and correlated signals on the same measurements
and compares their recovery.

## Null

A null signal asks how often a procedure reports features when there is nothing to find.
`NullTruth()` sets the generating score to zero on every row, so no feature affects the outcome
and every selected feature is a false selection. Count how many features are selected, and how
often the same ones recur across repeated datasets: recurrence shows which features the
procedure tends to pick when there is nothing to find. Recovery cannot be computed, because there
are no generating features to recover.

## Sparse

A sparse signal asks whether a method finds the few individual features that generate the
outcome, rather than their correlates. `SparseTruth` takes either the generating features by
name, with optional non-zero `weights` (default 1.0 each), or a `count` of features for
PyPlasmode to choose, no two of which are near-copies
([Signal construction](methods.md#signal-construction)).

```python
import numpy as np
import pyplasmode as ppm

a = np.array([-1, -1, 1, 1])
b = np.array([-1, 1, -1, 1])
names = ("A", "A_proxy", "B", "C")
population = ppm.Population(np.column_stack((a, a + b / 2, b, a * b)), names)
ranking = ("A_proxy", "A", "B", "C")

sparse = ppm.SparseTruth(features=("A", "B"))
sparse_truth = ppm.materialize_truth(sparse, population, seed=3)
print(ppm.evaluate_ranking(ranking, sparse_truth, depths=(2,)).at_depth[0].recall)
# 0.5

auto_sparse = ppm.SparseTruth(count=2, max_absolute_correlation=0.7)
print(ppm.materialize_truth(auto_sparse, population, seed=3).direct_features)
# ('C', 'B')
```

The examples on this page share a four-row population: every combination of A and B at −1 and 1, a
correlate `A_proxy` (A plus half of B, correlation 0.89 with A) and `C`, the product of A and B.
`ranking` stands in for a model's importance ranking, most important first. `materialize_truth`
computes the generating score and recovery targets without simulating outcomes. The top two features
hold A but not B, so `evaluate_ranking` reports a recovery of 0.5 (the API calls it `recall`).

Here the seeded search accepted C and B. It could never accept both A and A_proxy, because
their correlation of 0.89 exceeds the limit.

## Correlated

A correlated signal asks whether a method finds a generating feature or settles for one of its
*correlates*, measured features strongly correlated with it. Put each generating feature and its
correlates in a named `FeatureSet` and pass the collection as `feature_sets` to `generate`. The
groups must not share members, so that a selected feature counts towards one group only.

`CorrelatedTruth(group_names=...)` has two forms:

- **One generating feature per group** (the default). The generating feature is called the
  group's *sentinel* (`sentinels=` in code); the other members only correlate with it. This asks
  whether a method picks the right member. Omit `sentinels` to draw one member of each group at
  random.
- **Group means** (`signal_form="group_means"`). Every member contributes its standardised
  value to the group's average, so no single member is the right one. This asks whether a
  method finds the group.

Either form takes one weight per group (default 1.0).

```python
correlated_sets = ppm.FeatureSets((ppm.FeatureSet("A_group", ("A", "A_proxy")),))
correlated = ppm.CorrelatedTruth(group_names=("A_group",), sentinels=("A",))
outcome = ppm.ContinuousOutcome(mean=0.0, standard_deviation=1.0, variance_explained=0.25)
correlated_sample = ppm.generate(
    population,
    truth=correlated,
    feature_sets=correlated_sets,
    outcome=outcome,
    sample_size=4,
    sampling_method="without_replacement",
    seed=3,
)
print(ppm.evaluate_groups(ranking, correlated_sample.truth, depths=(1,)).at_depth[0].group_recall)
print(ppm.evaluate_ranking(ranking, correlated_sample.truth, depths=(1,)).at_depth[0].recall)
# 1.0
# 0.0
```

Only the recovery targets matter here, so `generate` draws each row once with an arbitrary
continuous outcome. A generates the outcome and A_proxy is
its correlate. The ranking's top feature is A_proxy, so
the two evaluators disagree. `evaluate_groups` counts a group as found when any member is
selected, and reports 1.0. `evaluate_ranking` asks for the sentinel itself, and reports 0.0.
Reporting both shows how often a method finds the right group but the wrong member. Under group
means there is no sentinel; use `evaluate_groups`, or `evaluate_module` for the share of
generating weight the selection covers.

## Distributed

A distributed signal asks how much of a signal a selection captures when many features each
carry part of it, as the members of a biological pathway might, so recovery is measured as a
share of the total. `ModuleTruth(set_name=...)` adds the standardised values of every member of
a named feature set, each multiplied by its weight (default 1.0, in set order); pass the set in
`feature_sets` to `generate`.

```python
module_sets = ppm.FeatureSets((ppm.FeatureSet("pathway", ("A", "B", "C")),))
module = ppm.ModuleTruth(set_name="pathway", weights=(2.0, -1.0, 1.0))
module_sample = ppm.generate(
    population,
    truth=module,
    feature_sets=module_sets,
    outcome=outcome,
    sample_size=4,
    sampling_method="without_replacement",
    seed=3,
)
coverage = ppm.evaluate_module(ranking, module_sample.truth, depths=(2,))
print(coverage.at_depth[0].weighted_coverage)  # 0.5
```

The weights give A twice the weight of B or C, and make the score fall as B rises.
`evaluate_module` divides the absolute weight of the selected generating features by the total
absolute weight, so a sign does not reduce credit: B's weight of −1 counts as 1. The top two
features are A_proxy and A; A carries 2 of the 4 units of weight, so the coverage is 0.5.

Coverage gives A_proxy no credit, although it carries much of A's information. To measure how well
the selected features predict the generating score on held-out rows, whichever features they are,
use [`evaluate_signal_reconstruction`](evaluation.md#signal-reconstruction). `evaluate_ranking`
raises `ValueError` for this signal, because it has no individual generating features to find.

## Interaction

An interaction signal asks whether a method detects features that matter only in combination, such
as a risk that rises only when two measurements are both high. With the product form, each feature
on its own can show no association with the outcome, so a method that judges features one at a time
can miss both. `InteractionTruth(features=("A", "B"))` builds the score in one of two ways:

- `operation="product"` (the default) multiplies the two standardised values, so the effect of
  one feature is proportional to the value of the other, and reverses when the other is below its
  mean.
- `operation="threshold"` gives 1 when both standardised values are above `threshold` (in
  standard deviations, default 0.0) and 0 otherwise, so the score takes two values: one for rows
  above the cutoff on both features, one for all other rows.

```python
interaction = ppm.InteractionTruth(features=("A", "B"), operation="threshold", threshold=0.0)
interaction_truth = ppm.materialize_truth(interaction, population, seed=3)
selection = ppm.nomination_from_scores(names, (3, 0, 1, 1), depth=2)
interaction_recovery = ppm.evaluate_fractional_recovery(
    selection,
    interaction_truth,
    estimand="interaction",
)
print(interaction_recovery.value)  # 0.5
```

A cutoff of 0.0 requires both features to be above their means. Recovery of an interaction is
the probability that the selection contains both features. `nomination_from_scores` takes
importance scores (3, 0, 1, 1 for A, A_proxy, B and C) and keeps the top two (the API calls a
selection a *nomination*). A is selected; B and C tie for the second place, so each is selected
with probability 0.5. `evaluate_fractional_recovery` averages over the two ways of breaking the
tie, and only one completes the pair, so recovery is 0.5.

If every row or no row clears the threshold on both features, the score is constant and
generation raises `ValueError`, because a constant score cannot be standardised. Choose a
threshold that some rows clear on both features and others do not: lower it if no row clears it,
raise it if every row does.

## Custom signal

A custom signal answers a question the built-in structures cannot, such as whether a method
copes with a response that flattens at extreme measurements. Wrap a function in
`CustomTruth(function=...)`. PyPlasmode calls it with three arguments: the sampled matrix
(read-only), its column names and a seeded NumPy random generator. It must return a
`CustomSignal` whose `signal` has one value per row. PyPlasmode then standardises it, as it does
a built-in score.

```python
def saturation(X, feature_ids, rng):
    values = X[:, feature_ids.index("A")]
    return ppm.CustomSignal(signal=np.tanh(values), direct_features=("A",))


custom = ppm.CustomTruth(function=saturation)
custom_sample = ppm.generate(
    population,
    truth=custom,
    outcome=outcome,
    sample_size=4,
    sampling_method="without_replacement",
    seed=3,
)
print(ppm.evaluate_ranking(ranking, custom_sample.truth, depths=(2,)).at_depth[0].recall)
# 1.0
```

`saturation` makes A the only generating feature through `tanh`, which flattens beyond about two
units. Here A takes only the values −1 and 1, so the standardised score equals A; on real data the
curve flattens in the tails. Declaring A in `direct_features` lets `evaluate_ranking` score the
ranking, and A is in the top two.

The score must be finite on every row and must vary, or generation raises `ValueError`. The
matrix still holds its missing values, so fill them in your function before using a column
that has any.

PyPlasmode records only the recovery targets you declare: `direct_features` for
`evaluate_ranking`, `groups` for `evaluate_groups`, `weights` for `evaluate_module` and
`graph_region` for `evaluate_graph_region`.

PyPlasmode calls the function twice with the same inputs and seed and raises `ValueError` if the
scores or declared targets differ, because a dataset that cannot be regenerated from its seed cannot
be reproduced; draw any randomness from the `rng` argument. Return a new array on each call, so the
second call cannot overwrite the first result. `generate_partitioned` does not accept a custom
signal and raises `ValueError`.
