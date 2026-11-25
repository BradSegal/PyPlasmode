# Generating signals

The *signal* (`truth=` in code) is what you specify: which features generate the outcome
and in what structure. It yields the *generating score* (`truth.signal`), one number per
row, and the features or groups a selection procedure should recover. Pair it with an
[outcome model](outcomes.md), which sets how strong it is.

![Five signal structures and their corresponding recovery targets](assets/concepts/signal-recovery.png)

In the diagram, "drivers" are generating features and "latent disease state" is the generating
score. Construct and evaluate sparse and correlated signals on the same measurements in the
[tutorial](tutorial.md).

## Null

`NullTruth()` sets the generating score to zero, so no feature affects the outcome through
the signal. Use it to measure how often
a procedure selects features when no association has been added: any selected features
measure false selection. Record the number selected and their recurrence across runs;
recovery is undefined when there are no generating features.

## Sparse

Use a sparse signal to test whether a model finds a few individual generating features.
`SparseTruth` takes either `features` with optional finite, non-zero `weights` (default 1.0
per feature) or `count` to select non-constant features with pairwise absolute correlation at most
`max_absolute_correlation` (default 0.7). Use `materialize_truth` to compute the generating
score and recovery targets without simulating outcomes.

```python
import numpy as np
import pyplasmode as ppm

a = np.array([-1, -1, 1, 1])
b = np.array([-1, 1, -1, 1])
names = ("A", "A_proxy", "B", "C")
population = ppm.Population(np.column_stack((a, a + b / 2, b, a * b)), names)
ranking = ("A_proxy", "A", "B", "C")

sparse = ppm.SparseTruth(features=("A", "B"))
auto_sparse = ppm.SparseTruth(count=2, max_absolute_correlation=0.7)
sparse_truth = ppm.materialize_truth(sparse, population, seed=3)
print(ppm.evaluate_ranking(ranking, sparse_truth, depths=(2,)).at_depth[0].recall)
# 0.5 (A is the only generating feature in the top two)
```

`evaluate_ranking` reports the fraction of generating features found as `recall`.
If the seeded search falls short of `count`, generation raises `ValueError`.

## Correlated

Use a correlated signal to test whether a model finds a generating feature or one of its
*correlates*, measured features strongly correlated with it. Put each group in a named
`FeatureSet`. Pass the collection as `feature_sets` to `generate`.

`CorrelatedTruth(group_names=..., sentinels=...)` uses one generating feature, called a
*sentinel*, per group. Omit `sentinels` to draw one member at random from each group. Set
`signal_form="group_means"` to average the standardised members of each group. Either form
takes one `weights` value per group (default 1.0). The named groups must be disjoint.

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
# 1.0 (A_proxy recovers the only group)
```

`evaluate_groups` counts a group as found when any member is selected. Use `evaluate_ranking`
to score the sentinels or `evaluate_module` to score the generating weight covered by a
selection under `group_means`.

## Distributed

Use a distributed signal when all members of a feature set contribute to the generating
score. `ModuleTruth(set_name=..., weights=...)` combines standardised members in set order.
Each weight defaults to 1.0. Pass the named set in `feature_sets` to `generate`.

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
print(coverage.at_depth[0].weighted_coverage)  # 0.5 (A carries 2 of 4 absolute weight)
```

`evaluate_module` divides the selected absolute generating weight by the total absolute
generating weight. `evaluate_signal_reconstruction` measures how well the panel predicts the
generating score on held-out rows. `direct_features` is empty, so `evaluate_ranking` raises
`ValueError`.

## Interaction

Use `InteractionTruth(features=("A", "B"))` when two features jointly generate the outcome.
The default `operation="product"` multiplies their standardised values. With
`operation="threshold"`, both standardised values must strictly exceed `threshold`, in
standard deviations from each feature's mean (default 0.0).

```python
interaction = ppm.InteractionTruth(features=("A", "B"), operation="threshold", threshold=0.0)
interaction_truth = ppm.materialize_truth(interaction, population, seed=3)
selection = ppm.nomination_from_scores(names, (3, 0, 1, 1), depth=2)
interaction_recovery = ppm.evaluate_fractional_recovery(
    selection,
    interaction_truth,
    estimand="interaction",
)
print(interaction_recovery.value)  # 0.5 (A is selected and B ties with C for the other slot)
```

With `estimand="interaction"`, recovery is the probability of selecting both features
when tied slots are filled at random. If every row or no row clears the joint threshold,
the signal is constant and generation raises `ValueError`.

## Custom signal

Use `CustomTruth(function=...)` to define your own generating score, such as a response
that flattens at extreme measurements. The callback `(X, feature_ids, rng) -> CustomSignal`
receives the read-only resampled matrix, its column names and a seeded NumPy generator.
Return a one-dimensional `signal` array with one finite value per row and variation across
rows. PyPlasmode standardises it.

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
# 1.0 (A is the only generating feature and is in the top two)
```

Set `direct_features` (a tuple of names) for `evaluate_ranking`, `groups` (a tuple of
`RecoveryGroup` objects) for `evaluate_groups`, or `weights` (a tuple of `FeatureWeight`
objects) for `evaluate_module`. Set `graph_region` to a tuple of feature names and pass it as
`region` with your graph to `evaluate_graph_region`. You can also set a text `description`.

PyPlasmode calls the function twice with identical inputs and seed. The score and targets
must agree across calls. Return a fresh score array each time so the second call cannot
overwrite the first result. `generate_partitioned` requires a built-in signal structure.
