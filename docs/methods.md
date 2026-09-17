# Methods

## Generation process

A plasmode combines empirical feature data with a known outcome-generating mechanism.
This lets you measure whether a method predicts the outcome and whether it recovers the
features responsible for that outcome. See [Gadbury et al.](references.md) for the plasmode
approach and [Morris et al.](references.md) for organising simulation studies with the ADEMP framework.

![Observed measurements supply realistic features; a specified signal supplies the recovery target](assets/concepts/plasmode-workflow.png)

In the biomarker example above, predictions are compared with the generated outcomes and
selected features with the known signal. The same comparison applies to any numeric feature
matrix: prediction accuracy and feature recovery measure different parts of the procedure.

Generation has three steps:

1. Sample complete rows from the source matrix. The default is sampling with replacement;
   use `sampling_method="without_replacement"` to include each source row at most once.
2. Construct and standardise the chosen signal on the sampled rows. Missing values in signal
   features are median-imputed for this calculation; returned `X` retains its missingness.
3. Calibrate the outcome model to the requested marginal targets and draw the outcomes.

You can then apply your usual preprocessing, model fitting, hyperparameter search and
explanation tools. PyPlasmode's evaluators take the resulting predictions and feature selections.

## Correlation and missingness

Complete-row resampling draws from the joint empirical distribution, retaining observed
feature relationships and missingness patterns. The source population therefore determines
the combinations available to the simulation. The signal and outcome specification determine
the association being studied.

## Training, validation and test samples

`partition_population` splits source rows before resampling. `generate_partitioned` samples
within those partitions and returns the original row indices. A source row can occur
repeatedly within a partition when sampling with replacement, but cannot occur in both
training and evaluation. Automatic selection of sparse features or generating group members
uses training source rows; the same features are then used in every partition.

Each row is treated as a sampling unit. If several rows belong to one person or other
cluster, arrange the split at that level before generation. Row-level splitting alone
does not keep related observations together.

The generated partitions share one signal scale and outcome calibration, computed over their
combined covariates. This gives them the same generating mechanism. Model preprocessing is
a separate step fitted on training data. Changing the evaluation sample can change the
shared calibration; a transfer study requiring a fixed generating model across populations
needs a separately specified mechanism.

## Signal construction

Each signal feature is median-imputed and standardised to mean zero and population standard
deviation one. Features outside an explicitly specified signal are left unprocessed.
Signal features need observed variation: constant or entirely missing columns are excluded
from automatic selection and raise `ValueError` when explicitly selected.

For an additive signal, each selected feature contributes its standardised value multiplied
by a weight. Suppose features A and B have weights 2 and -1. An observation with values
`A=1` and `B=0.5` has score `s = 2*1 - 0.5 = 1.5`. Increasing A raises the score;
increasing B lowers it.

Let A and A_proxy be closely correlated features. A sparse signal can depend
on A alone. A correlated-group signal can use that same contributor while also recording
A_proxy as an acceptable substitute for group recovery. A distributed signal instead uses
both measurements, such as their average. The [tutorial](tutorial.md) constructs these
alternatives with the same population and compares their recovery targets.

| Signal | How to construct the score | What it tests |
| --- | --- | --- |
| Sparse | Add the weighted values of a few selected features. | Recovery of individual contributors |
| Correlated sentinel | Choose one contributor per group and add their weighted values. | Contributor recovery versus recovery of substitutes |
| Distributed module | Add weighted contributions from all module members. | Recovery of a shared signal |
| Group mean | Average features inside each group, then combine the group means. | Recovery of group-level information |
| Product | Multiply two standardised feature values. | Dependence on their combination |
| Joint threshold | Return one when both values exceed their thresholds, otherwise zero. | A relationship requiring both conditions |

The completed score is standardised as `z = (s - mean(s)) / sd(s)` before generating the
outcome. An odds ratio of 2 therefore means a doubling of the odds for a one-standard-deviation
increase in the combined score, rather than in each feature separately.
The [outcome guide](outcomes.md) explains how this score becomes an observed response.

`SparseTruth(count=k)` uses a seeded greedy search with a maximum absolute pairwise Pearson
correlation (default 0.7). Set this threshold for your design, or supply feature identities
directly when the hypothesis concerns particular features.

## Numerical calibration and random variation

Calibration matches the expected outcome across the sampled rows to your requested target.
For example, a binary probability of 0.15 means an average event probability of 15%, not
exactly 15% events in every draw. SciPy's bracketed root finder solves these expectation
equations; the [outcome guide](outcomes.md#calibration-equations) gives them for each family.

Binary and ordinal intercepts use a `[-50, 50]` logit bracket; hazard calibration expands
a non-negative bracket geometrically. A target outside the solver's range raises
`CalibrationError`.

Sampled outcomes are checked against their expected summaries using six sampling standard
errors, with a six-observation floor for probabilities. These diagnostic tolerances account
for sampling variation. A rare draw can exceed them and raise `RealizationError`; include
such failures in replicate summaries. Use independent replicates to estimate performance
and Monte Carlo uncertainty.
