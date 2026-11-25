# Methods

## Generation process

A plasmode combines empirical feature data with outcomes generated from a known signal.
This lets you measure whether a method predicts the outcome and whether it recovers the
generating features, those used to generate that outcome.
[Gadbury et al.](references.md#plasmode-simulation)
illustrate the plasmode approach; [Schreck et al.](references.md#plasmode-simulation) describe
the combination of covariate resampling and outcome generation used here. For organising
the comparison, [Morris et al.](references.md#simulation-studies) introduce the ADEMP framework.

![Observed measurements supply realistic features; a specified signal supplies the recovery target](assets/concepts/plasmode-workflow.png)

In the biomarker example, the figure compares predictions with generated outcomes and selected
features with the known signal. The same comparison applies to any numeric feature matrix.
Prediction accuracy and feature recovery measure different aspects of the procedure.

Generation has three steps:

1. Sample complete rows from the source matrix. The default is sampling with replacement;
   use `sampling_method="without_replacement"` to include each source row at most once.
2. Construct and standardise the generating score on the sampled rows
   ([Signal construction](#signal-construction)). Returned `X` retains its missingness.
3. Calibrate the outcome model to the requested marginal targets and draw the outcomes.

You can then apply your usual preprocessing, model fitting, hyperparameter search and
explanation tools. PyPlasmode's evaluators take the resulting predictions and feature
selections.

## Correlation and missingness

Complete-row resampling samples from the joint empirical distribution, preserving relationships
between features and missingness patterns. The source population determines which combinations
are possible in the simulation. The signal and outcome specification determine the association
being studied.

## Training, validation and test samples

`partition_population` splits source rows before resampling. `generate_partitioned` samples
within these partitions and returns original row indices in `training_source_indices`,
`validation_source_indices` and `evaluation_source_indices`. Each sample's `source_indices` are
indices within its partition. Sampling with replacement can repeat a row within a partition. A
source row cannot occur in both training and evaluation. Automatic selection of sparse
generating features uses training source rows. An automatically selected generating member is
drawn at random from its group and checked on training source rows. The same features are used
for all partitions.

Each row is a sampling unit. If you have multiple rows per person or cluster, split at that
level before generation. Row-level splitting alone does not keep related observations together.

The generated partitions share one scaling of the generating score and one outcome
calibration, computed over their combined covariates, so the same signal generates the
outcome in each. Model preprocessing is a separate step fitted on training data. Changing
the evaluation sample can change the shared calibration; a transfer study requiring a fixed
generating model across populations needs that model specified separately.

## Signal construction

The generating score (`truth.signal`) is the number computed for each row from the
generating features. The rules below construct it for each signal structure defined in
[Generating signals](truths.md).

For the built-in signal structures, each generating feature is median-imputed and standardised
to mean zero and population standard deviation one. Features outside an explicitly specified
signal are left unprocessed. Generating features need observed variation: constant or entirely
missing columns are excluded from automatic sparse selection and raise `ValueError` when
chosen as a generating feature.

For an additive signal, each generating feature contributes its standardised value multiplied
by a weight. Suppose features A and B have weights 2 and -1. An observation with values
`A=1` and `B=0.5` has score `s = 2*1 - 0.5 = 1.5`. Increasing A raises the score;
increasing B lowers it.

A *sentinel* is what the API calls a group's generating feature. The other members are its
*correlates*.

| Signal structure | How to construct the score | What it tests |
| --- | --- | --- |
| Sparse | Add the weighted values of a few selected features. | Recovery of individual generating features |
| Correlated, sentinels | Choose one generating feature per group and add their weighted values. | Recovery of generating features versus recovery of their correlates |
| Distributed | Add weighted contributions from all module members. | Recovery of a shared signal |
| Correlated, group means | Average features inside each group, then combine the group means. | Recovery of group-level information |
| Interaction, product | Multiply two standardised feature values. | Dependence on their combination |
| Interaction, joint threshold | Return one when both standardised values exceed the threshold, otherwise zero. | A relationship requiring both conditions |

The completed score is standardised as `z = (s - mean(s)) / sd(s)` before generating the
outcome; a null score stays zero. An odds ratio of 2 therefore means a doubling of the odds for a one-standard-deviation
increase in the generating score, rather than in each feature separately.
The [outcome guide](outcomes.md) explains how this score becomes an observed response.

`SparseTruth(count=k)` uses a seeded greedy search with a maximum absolute pairwise Pearson
correlation (default 0.7). Set this threshold for your design, or supply feature identities
directly when the hypothesis concerns particular features.

## Numerical calibration and random variation

Calibration matches the expected outcome across sampled rows to your target. A binary
probability of 0.15 means an average event probability of 15%. The fraction of events can vary
across draws. The [outcome guide](outcomes.md#calibration-equations) gives each family's
expectation equation. SciPy's bracketed root finder solves equations without a closed-form
solution.

Binary and ordinal intercepts use a logit bracket of at least `[-50, 50]`, widened as needed
for the target and offsets; hazard calibration expands a non-negative bracket geometrically.
A target outside the solver's range raises `CalibrationError`.

Sampled outcomes are checked against their requested targets, with a tolerance of six standard
errors and a six-observation floor for probabilities. A rare draw can exceed these
tolerances and raise `RealizationError`; include such failures in replicate summaries. Use
independent replicates to estimate performance and Monte Carlo uncertainty.

## Discrimination and repeated selections

The [metrics guide](metrics.md) defines selection stability, consensus rankings and AUC (area
under the ROC curve) under censoring. The [calibration guide](calibration.md) explains how a
target AUC of the generating score determines an odds or hazard ratio. Expected and realised AUC
are reported separately. For a rare outcome there may be substantial realised sampling
variation.
