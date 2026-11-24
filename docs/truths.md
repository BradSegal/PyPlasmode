# Generating signals

The *signal* (`truth=` in code) is what you specify: which features generate the outcome
and in what structure. It yields the *generating score* (`truth.signal`), one number per
row, and the features or groups a selection procedure should recover. Pair it with an
[outcome model](outcomes.md), which sets how strong it is.

![Five signal structures and their corresponding recovery targets](assets/concepts/signal-recovery.png)

The diagram pairs each signal you can create with what you ask the model to recover.
It labels generating features "drivers" and the generating score a "latent disease state".
The [tutorial](tutorial.md) constructs sparse and correlated signals on the same
measurements and shows the corresponding evaluations.

## Null

`NullTruth()` sets the generating score to zero, so no feature affects the outcome through
the signal. Use it to measure how often
a procedure selects features when no association has been added: any selected features
measure false selection. Record the number selected and their recurrence across runs;
recovery is undefined when there are no generating features.

## Sparse

A sparse signal has a few generating features. `SparseTruth(features=(...))` combines named
features with finite, non-zero weights. `SparseTruth(count=k)` selects features
automatically under a pairwise correlation threshold. Measure recovery of these generating
features with `evaluate_ranking`.

## Correlated

A correlated signal separates finding a generating feature from finding one of its
*correlates*: measured features strongly correlated with it. `CorrelatedTruth` uses disjoint
named `FeatureSet` groups, which you supply, for example from correlations or pathway
annotations. With the default `signal_form="sentinels"`, one feature from each group (its
*sentinel*) enters the generating score; the other members count as its correlates.
`evaluate_ranking` measures recovery of the generating features, while `evaluate_groups`
measures recovery of any member of each group.

With `signal_form="group_means"`, the generating score combines within-group means. Evaluate
it with group recovery or weighted recovery. It has no individual sentinels:
`direct_features` is empty and each group's `direct_feature` is `None`.

## Distributed

A distributed signal spreads the generating score across all members of a feature set.
`ModuleTruth` combines them with specified weights. Evaluate the selected panel by its
weighted recovery, the share of the absolute generating weight it covers, or its ability to
reconstruct the generating score on held-out data.

## Interaction

An interaction signal makes the outcome depend on a combination of two features.
`InteractionTruth` combines them as a product or a joint threshold. Interaction recovery
credits selection of both members.

## Custom signal

`CustomTruth` covers a signal structure that the built-in classes do not. Its function
receives the resampled matrix, feature identities and a seeded NumPy generator, and returns a
`CustomSignal` containing the generating score, a description and its exact, group, weighted
or graph recovery targets. The
[custom-signal example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/custom_signal.py)
uses a saturating relationship: a feature changes the score strongly near its centre, but
its effect flattens at high or low measurements.

Inputs are read-only, and the function should return the same finite, nonconstant score and
targets for identical inputs and seed. PyPlasmode checks this by calling it twice and
comparing the results, so return a new array on each call.
Custom functions work with `generate`. For `generate_partitioned`, use a built-in signal
structure so the same signal can be applied across partitions.
