# Methods

## Generation process

To test whether a method finds the features that drive an outcome, you need data in which you
know the answer. Real measurements have realistic distributions, correlations and missing
values, but nobody knows which features drive a real outcome. A plasmode keeps your real feature
rows and replaces the outcome with one generated from features you choose, the *generating
features*. You can then measure whether a method predicts the outcome and whether it recovers
the generating features.
[Gadbury et al.](references.md#plasmode-simulation) illustrate the plasmode approach;
[Schreck et al.](references.md#plasmode-simulation) describe the combination of row resampling
and outcome generation used here; [Morris et al.](references.md#simulation-studies) give the
ADEMP framework for planning such a simulation study.

![Observed measurements supply realistic features; a specified signal supplies the recovery target](assets/concepts/plasmode-workflow.png)

Each call to `generate` takes three steps:

1. **Draw whole rows from your matrix.** By default rows are drawn with replacement, so samples
   vary from draw to draw as bootstrap samples do. With
   `sampling_method="without_replacement"` each source row appears at most once; use it when a
   repeated row would distort what you measure, for example when you split one sample yourself
   and copies of a row could land on both sides.
2. **Compute the generating score** from the generating features on the sampled rows
   ([Signal construction](#signal-construction)). The returned matrix `X` is left as drawn,
   missing values included, so your preprocessing meets the same gaps it would meet on the
   real data.
3. **Calibrate the outcome model and draw one outcome per row**
   ([Numerical calibration](#numerical-calibration-and-random-variation)).

Fitting, tuning and explaining a model stay in your own tools. PyPlasmode's evaluators then
take the predictions and feature selections those tools produce.

## Correlation and missingness

Feature selection is hard on real data because measurements are correlated: a model can pick
a correlate of a generating feature in place of the feature itself. Missing values also follow
patterns that depend on other measurements. Each sampled row is a whole source row, so a sample
keeps the correlations and missingness patterns of your data exactly, and the method you test
faces the problem it would face on the real measurements. The consequence is that a sample
contains only combinations of values that occur in the source.

## Training, validation and test samples

A model must be tested on rows it never saw while being fitted or tuned. If you draw one sample
with replacement and split it afterwards, copies of the same source row can land in both the
training and the test sample, and the model is tested on rows it has already seen. To prevent
this, split the source rows first with `partition_population`, then draw each sample from its
own pool with `generate_partitioned`. A source row then belongs to one pool: it can repeat
within the training sample but never appears in the test sample. The API calls the test sample
`evaluation`.

```python
import numpy as np
import pyplasmode as ppm

population = ppm.Population(np.arange(12.0).reshape(-1, 1), ("A",))
partition = ppm.partition_population(
    population, training_fraction=0.5, validation_fraction=0.25, seed=0
)
samples = ppm.generate_partitioned(
    partition,
    truth=ppm.SparseTruth(features=("A",)),
    outcome=ppm.BinaryOutcome(probability=0.5, odds_ratio=2),
    sample_sizes=ppm.PartitionSampleSizes(training=6, validation=3, evaluation=3),
    sampling_method="without_replacement",
    seed=0,
)
print(samples.evaluation.source_indices[0], samples.evaluation_source_indices[0])  # 1 8
print(samples.evaluation.X[0])  # [8.]
```

The population has twelve rows, and the value of A in each is its row number, so you can see
where a sampled row came from. The fractions 0.5 and 0.25 split the rows into pools of six, three
and three. Each sample records where its rows came from at two levels. `samples.evaluation.source_indices` counts within the evaluation pool: the first
test row is row 1 of `partition.evaluation`. `samples.evaluation_source_indices` counts within
your original `Population`: the same row is source row 8, which its value confirms. Use the
second to link sampled rows back to your own records. Training and validation have the same two
fields.

`partition_population` splits individual rows. If one person or cluster contributes several
rows, those rows resemble each other and leak between pools in the same way as copies. Split
them by person or cluster yourself and build the `PopulationPartition` from your own row indices.

When PyPlasmode chooses the generating features for you, with `SparseTruth(count=...)` or with
`CorrelatedTruth` and no `sentinels`, it chooses them on the training pool only, so that no test
row influences the choice. All three samples then use the same generating features. A
correlated signal draws one member of each group at random; if that member is constant or
entirely missing in the training pool, generation raises `ValueError`, because such a feature
cannot carry a signal.

The generating score is standardised, and the outcome calibrated, on the training, validation and
test rows together, so that one generating model produces all three samples. Two consequences
follow:

- **Targets hold on the combined rows.** With `probability=0.15`, the event fraction across all
  three samples is close to 15%, and the fraction in one sample can differ from it.
- **The test rows are part of the generating model.** Drawing different test rows changes the
  score's scaling and the outcome's baseline for the training rows as well.

This scaling belongs to the simulation. Fit your model's own preprocessing on the training
sample only, as on real data, so that no test information reaches the model.

To compare a method across populations, for example two cohorts, call `generate` once per
population with the same effect, such as the same odds ratio. Each call standardises the score
and calibrates the baseline on its own rows, so the effect is per standard deviation of each
population's own score. PyPlasmode has no option to apply one fixed generating model to several
populations: to do that, compute the score and draw the outcomes in your own code. With
`target_auc`, each call also solves its own effect
([AUC calibration](calibration.md#what-is-solved)).

## Signal construction

The outcome is drawn from the generating score (`truth.signal` in code), the number computed for
each row from the generating features. Every effect you set, such as an odds ratio, is per
standard deviation of this score. PyPlasmode builds the score so that this unit means the same
whatever the features' units and whichever signal structure you choose. For the built-in
signals it takes four steps:

1. **Fill missing values** of each generating feature with that feature's median, so the score
   is defined on every row. Only the score uses the filled values; `X` keeps its missing values.
2. **Standardise each generating feature** to mean zero and population standard deviation one,
   so a weight measures a feature's contribution in standard deviations, not in its own units.
3. **Combine the features** as the signal structure specifies (table below).
4. **Standardise the combined score** to mean zero and population standard deviation one, so an
   odds ratio of 2 doubles the odds of the event for each one-standard-deviation increase in
   the score. A null score stays zero.

For example, with weights 2 and −1, standardised values A = 1 and B = 0.5 give
`2*1 - 0.5 = 1.5` at step 3. Step 4 then rescales the score, so only the ratio of the weights
matters: weights (4, −2) give the same score as (2, −1).

A generating feature you name that is constant or entirely missing raises `ValueError`, because
it cannot be standardised and would contribute nothing. Automatic sparse selection skips such
features.

In the table, a *sentinel* is the one generating feature taken from a group of correlated
features, its *correlates* are the other members of the group, and a *module* is a named
`FeatureSet` whose members all contribute. [Generating signals](truths.md) explains when to
choose each structure.

| Signal structure | How to construct the score | What it tests |
| --- | --- | --- |
| Sparse | Add the weighted values of selected features. | Recovery of individual generating features |
| Correlated, sentinels | Add the weighted values of one generating feature per group. | Recovery of generating features and their correlates |
| Distributed | Add weighted values from all module members. | Recovery of a shared signal |
| Correlated, group means | Average standardised values within each group. Apply group weights and sum the means directly. | Recovery of group-level information |
| Interaction, product | Multiply two standardised values. | Dependence on their combination |
| Interaction, joint threshold | Give one if both standardised values are strictly greater than `threshold` (default `0.0`), otherwise zero. | A relationship requiring both conditions |

`SparseTruth(count=k)` picks k generating features for you. It visits the features in an order
set by the seed and accepts a feature only if its absolute Pearson correlation with every feature
already accepted is at most `max_absolute_correlation` (default 0.7). The limit stops two
generating features from being near-copies; otherwise a model that selected one would be scored
as missing the other although the two carry the same information. If fewer than k features pass,
generation raises `ValueError`: ask for fewer features, raise the limit or name the features
yourself.

## Numerical calibration and random variation

Each simulated dataset should represent the scenario you specified. PyPlasmode checks this twice.

First, calibration sets the model's intercept or baseline so that the *expected* outcome over
the sampled rows equals your target exactly. With `probability=0.15`, the rows' event
probabilities average 15%. The drawn outcomes vary around that, so one dataset might contain
14.6% events. [Calibration equations](outcomes.md#calibration-equations) gives the model for
each outcome. If no intercept meets the target with the effect you set, generation raises
`CalibrationError`. For example, a risk ratio of 3 with `probability=0.4` pushes the
probabilities of the highest-scoring rows above one. Lower the risk ratio, or use an odds ratio,
whose probabilities cannot leave the range 0 to 1.

Second, after drawing, PyPlasmode compares each observed summary with its target and raises
`RealizationError` if they differ by more than about six standard errors. A correct draw almost
never fails this, so a failure marks a dataset that does not represent your scenario. With `p`
the target probability and `n` the number of rows, the tolerances are:

- **Probabilities** (the binary event fraction, each ordered category's fraction, and survival
  event and censoring risks at each specified time): the larger of `6*sqrt(p*(1-p)/n)` and
  `6/n`. The second term allows a difference of six events when events are so rare that the
  first term would be smaller.
- **Survival event risks** are checked on the drawn event times before censoring, so heavy
  censoring does not by itself cause a failure.
- **Continuous outcomes**: `6*sd/sqrt(n)` for the mean and `6*sd/sqrt(2*(n-1))` for the standard
  deviation, with `sd` the target standard deviation.
- **Counts**: six times the standard deviation of the total count, divided by total exposure.

When you repeat a simulation, catch `RealizationError` and count the failures. Dropping failed
draws silently makes a study look as though every draw succeeded.

```python
event_fractions = []
failure_count = 0
for seed in (1, 2, 3):
    try:
        replicate = ppm.generate(
            population,
            truth=ppm.NullTruth(),
            outcome=ppm.BinaryOutcome(probability=0.5, odds_ratio=1),
            sample_size=12,
            seed=seed,
        )
        event_fractions.append(replicate.outcome.values.mean())
    except ppm.RealizationError:
        failure_count += 1

summary = ppm.summarize_replicates(event_fractions, failure_count=failure_count)
print(summary.replicate_count, summary.failure_count)  # 3 0
```

The loop draws three datasets, keeps each event fraction, and counts a failed draw instead of
letting the error stop the study. All three pass here. `summarize_replicates` reports the
failure count next to the mean of the successful draws, so a reader of your results sees both.

## Discrimination and repeated selections

To set how hard the prediction task is, set `target_auc` on a binary or time-to-event outcome in
place of an odds or hazard ratio. The AUC (area under the ROC curve) is then that of the
generating score: the probability that it ranks a random case above a random control. No model
can expect to rank better than the true risks, so this AUC is the ceiling for every method you
test. PyPlasmode solves for the odds or hazard ratio that gives it
([AUC calibration](calibration.md)).

`outcome.calibration.oracle_auc` is the AUC the generating score is expected to reach (the API
calls it the *oracle* AUC). The AUC in one dataset differs from it by chance, most when events
are few. Measure it by scoring the generating score against the drawn outcomes:

```python
cohort = ppm.Population(np.random.default_rng(5).normal(size=(2_000, 3)), ("A", "B", "C"))
for rows in (2_000, 200):
    draw = ppm.generate(
        cohort,
        truth=ppm.SparseTruth(features=("A",)),
        outcome=ppm.BinaryOutcome(probability=0.1, target_auc=0.75),
        sample_size=rows,
        seed=4,
    )
    realised = ppm.evaluate_prediction(draw.outcome.values, draw.truth.signal, metric="roc_auc")
    oracle = round(draw.outcome.calibration.oracle_auc, 3)
    print(rows, draw.outcome.values.sum(), oracle, round(realised.value, 3))
# 2000 175 0.75 0.754
# 200 18 0.75 0.799
```

With 2,000 rows and 175 events the realised AUC is 0.754; with 200 rows and 18 events it is
0.799. Compare a model's AUC with the realised AUC of the generating score in the same dataset,
not with the target, because both share that dataset's luck.

When you refit a model on resampled data, its selected features change from fit to fit.
`selection_stability` measures how much the selections agree and `consensus_ranking` combines
them into one ranking. For time-to-event outcomes, `cumulative_dynamic_auc` measures AUC while
accounting for censoring. All three are in `pyplasmode.metrics` ([Metrics](metrics.md)).
