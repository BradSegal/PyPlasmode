# Methods

## Generation process

To test whether a method finds the features that drive an outcome, you need data in which you
know the answer, and a real outcome never tells you. A plasmode keeps your real feature rows, with
their distributions, correlations and missing values, and replaces the outcome with one generated
from features you choose, the *generating features*. You can then measure whether a method
predicts the outcome and whether it recovers the generating features.
[Gadbury et al.](references.md#plasmode-simulation) illustrate the approach;
[Schreck et al.](references.md#plasmode-simulation) describe the combination of row resampling
and outcome generation used here; [Morris et al.](references.md#simulation-studies) give the
ADEMP framework for planning such a simulation study.

![Observed measurements supply realistic features; a specified signal supplies the recovery target](assets/concepts/plasmode-workflow.png)

Each call to `generate` takes three steps:

1. **Draw whole rows from your matrix**, with replacement by default, so samples vary from draw
   to draw as bootstrap samples do. With `sampling_method="without_replacement"` each source row
   appears at most once; use it when a repeated row would distort what you measure, for example
   when you split one sample yourself and copies of a row could land on both sides.
2. **Compute the generating score** on the sampled rows
   ([Signal construction](#signal-construction)).
3. **Calibrate the outcome model and draw one outcome per row**
   ([Numerical calibration](#numerical-calibration-and-random-variation)).

Fit, tune and explain models in your own tools, then pass their predictions and feature
selections to PyPlasmode's evaluators.

## Correlation and missingness

Feature selection is hard on real data because measurements are correlated: a model can pick
a correlate of a generating feature in place of the feature itself. Missing values also follow
patterns that depend on other measurements. Each sampled row is a whole source row, so a sample
keeps the combinations of values and missing entries that occur in your data, and the method you
test faces the problem it would face on the real measurements. Which rows are drawn, and how
often, varies between draws, so a sample's correlations and missingness rates differ somewhat
from the source's.

## Training, validation and test samples

A model must be tested on rows it never saw while being fitted or tuned. If you split one sample
drawn with replacement, copies of a source row can land in both training and test. Instead, split
the source rows first with `partition_population`, then draw each sample from its own pool with
`generate_partitioned`: a source row can then repeat within one sample but never appears in two.
The API calls the test sample `evaluation`.

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

The value of A in each of the twelve rows is its row number, so you can see where a sampled row
came from. The fractions 0.5 and 0.25 give pools of six, three and three rows.
`samples.evaluation.source_indices` counts within the test pool: the first test row is row 1 of
`partition.evaluation`. `samples.evaluation_source_indices` counts within your original
`Population`: the same row is source row 8, as its value confirms. Use the second to link sampled
rows to your own records; training and validation have the same two fields.

`partition_population` splits individual rows. If one person or cluster contributes several
rows, those rows resemble each other and leak between pools in the same way as copies. Split
them by person or cluster yourself and build the `PopulationPartition` from your own row indices.

When PyPlasmode chooses the generating features for you, with `SparseTruth(count=...)`, or with
`CorrelatedTruth` in its default sentinel form and no `sentinels`, `generate_partitioned` chooses
them on the rows of the training pool only, so that no test row influences the choice. All three
samples then use the same generating features. The sentinel form draws one member of each group
at random; if that member is constant or entirely missing in the training pool, generation raises
`ValueError`, because such a feature cannot carry a signal.

The generating score is built, and the outcome calibrated, on the sampled training, validation
and test rows together, so that one generating model produces all three samples. The medians that
fill missing values, both standardisations and the intercept or baseline all come from the
combined rows. Two consequences follow:

- **Targets hold on the combined rows.** With `probability=0.15`, the event probabilities average
  15% across all three samples, and the fraction in one sample can differ from it.
- **The test rows are part of the generating model.** Drawing different test rows can change the
  training rows' generating scores and outcome probabilities.

This scaling belongs to the simulation. Fit your model's own preprocessing on the training
sample only, as on real data, so that no test information reaches the model.

To compare a method across populations, such as two cohorts, call `generate` once per population
with the same effect. Each call standardises the score and calibrates the baseline on its own
rows, so the effect is per standard deviation of that population's score. To apply one fixed
generating model to several populations, compute the score and draw the outcomes in your own
code. With
`target_auc`, each call also solves its own effect
([AUC calibration](calibration.md#what-is-solved)).

## Signal construction

Every effect you set, such as an odds ratio, is per standard deviation of the generating score
(`truth.signal` in code), the number computed for each row from the generating features.
PyPlasmode builds the score so that this unit means the same whatever the features' units and
signal structure. For the built-in
signals it takes four steps:

1. **Fill missing values** of each generating feature with that feature's median, so the score
   is defined on every row. Only the score uses the filled values; `X` keeps its missing values.
2. **Standardise each generating feature** to mean zero and population standard deviation one,
   so a weight measures a feature's contribution in standard deviations, not in its own units.
3. **Combine the features** as the signal structure specifies (table below).
4. **Standardise the combined score** to mean zero and population standard deviation one, so an
   odds ratio of 2 doubles the odds of the event for each one-standard-deviation increase in
   the score. A null score stays zero.

Because step 4 rescales the score, only the ratio of the weights matters: weights (4, −2) give
the same score as (2, −1).

A generating feature you name that is constant or entirely missing raises `ValueError`, because
it cannot be standardised. Automatic sparse selection skips such
features.

In the table, a *sentinel* is the one generating feature taken from a group of correlated
features, its *correlates* are the other members of the group, and a *module* is a named
`FeatureSet` whose members all contribute ([Generating signals](truths.md) explains when to
choose each).

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
already accepted, computed on the median-filled values, is at most `max_absolute_correlation`
(default 0.7). The limit stops two generating features from being near-copies; otherwise a model
that selected one would be scored as missing the other although the two carry the same
information. The search does not backtrack, so it can stop short of k even when k such features
exist. Generation then raises `ValueError`: ask for fewer features, raise the limit or name the
features yourself.

## Numerical calibration and random variation

Each simulated dataset should represent the scenario you specified. PyPlasmode checks this twice.

First, calibration sets the model's intercept or baseline so that the *expected* outcome over the
sampled rows equals your target, to numerical precision. With `probability=0.15`, the rows' event
probabilities average 15%. The drawn outcomes vary around that, so one dataset might contain 14.6%
events. If the model ([Calibration equations](outcomes.md#calibration-equations)) cannot meet the
target with the effect you set, generation raises `CalibrationError`. For example, a risk ratio of 3
with `probability=0.4` pushes the probabilities of the highest-scoring rows above one when the score
is normally distributed. Lower the risk ratio, or use an odds ratio, whose probabilities cannot
leave the range 0 to 1.

Second, after drawing, PyPlasmode compares each observed summary with its target and raises
`RealizationError` if they differ by more than about six standard errors. The expected value
already equals the target, so a failure means this one draw landed far from it by chance, which
is rare, not that your specification or data are wrong. With `p`
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

The loop draws three 12-row datasets with a null signal and a 50% event probability and counts any
failure instead of letting it stop the study. With 12 rows the tolerance is 0.87, wider than any
possible distance from 50%, so these draws cannot fail; the pattern is what matters.
`summarize_replicates` reports the failure count next to the mean of the successful draws, so a
reader of your results sees both.

## Discrimination and repeated selections

To set how hard the prediction task is, set `target_auc` on a binary or time-to-event outcome in
place of an odds or hazard ratio. The AUC (area under the ROC curve) is then the expected AUC of
the true risks: the probability that a random case has a higher risk than a random control, with
ties counted as half. PyPlasmode solves for the odds or hazard ratio that gives it
([AUC calibration](calibration.md)).

`outcome.calibration.oracle_auc` is this expected AUC (the API calls it the *oracle* AUC). The
AUC in one dataset differs from it by chance, most when events are few. Measure it by scoring the
true risks against the drawn outcomes. The generating score ranks rows as the true risks do when
the effect raises risk, as here; otherwise multiply it by `calibration["signal_coefficient"]`,
which is then negative.

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

The smaller draw, with 18 events, lands further from the target. Compare a model's AUC with the
realised AUC of the generating score in the same dataset, not with the target, because both share
that dataset's luck.

When you refit a model on resampled data, its selected features change from fit to fit.
`selection_stability` measures how much the selections agree and `consensus_ranking` combines
them into one ranking. For time-to-event outcomes, `cumulative_dynamic_auc` measures AUC while
accounting for censoring. All three are in `pyplasmode.metrics` ([Metrics](metrics.md)).
