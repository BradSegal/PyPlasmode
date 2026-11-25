# Methods

## Generation process

A plasmode lets you ask "does a method find the features that drive an outcome?" in data where
you know the answer, which a real outcome never tells you. It keeps your real feature rows,
with their distributions, correlations and missing values, and replaces the outcome with one
generated from features you choose, the *generating features*. You can then measure whether a
method predicts the outcome and whether it recovers the generating features.
[Gadbury et al.](references.md#plasmode-simulation) illustrate the approach;
[Schreck et al.](references.md#plasmode-simulation) describe the combination of row resampling
and outcome generation used here; [Morris et al.](references.md#simulation-studies) give the
ADEMP framework for planning such a simulation study.

![Observed measurements supply realistic features; a specified signal supplies the recovery target](assets/concepts/plasmode-workflow.png)

Each call to `generate`:

1. **draws whole rows from your matrix**, with replacement by default, so samples vary from draw
   to draw as bootstrap samples do;
2. **computes the generating score** on the sampled rows
   ([Signal construction](#signal-construction));
3. **calibrates the outcome model and draws one outcome per row**
   ([Numerical calibration](#numerical-calibration-and-random-variation)).

With `sampling_method="without_replacement"` each source row appears at most once; use it when a
repeated row would distort what you measure, such as when you split one sample yourself. Fit,
tune and explain models in your own tools, then pass their predictions and feature selections to
PyPlasmode's evaluators.

## Correlation and missingness

Feature selection is hard on real data because measurements are correlated. A model can pick a
correlate of a generating feature instead of the feature itself. Missing values also follow
patterns that depend on other measurements. Each sampled row is a whole source row, so a sample
keeps the combinations of values and missing entries that occur in your data, and the method you
test faces the problem it would face on the real measurements. Which rows are drawn and how
often they are drawn can vary between draws, so a sample's correlations and missingness rates
can differ a bit from the source's.

## Training, validation and test samples

Separate samples answer the question "how do you test a model on rows it never saw while it was
fitted or tuned?". If you split one sample drawn with replacement, copies of a source row can land
in both training and test. Instead:

1. `partition_population` splits the source rows into training, validation and test pools;
2. `generate_partitioned` draws each sample from its own pool, so a source row can repeat within
   one sample but never appears in two (the API calls the test sample `evaluation`);
3. generating features that PyPlasmode chooses for you (`SparseTruth(count=...)`, or
   `CorrelatedTruth` in its sentinel form without `sentinels`) are chosen on the training pool only,
   so that no test row influences the choice, and all three samples use them;
4. the generating score and the outcome model are built on the three samples together, so that
   one generating model produces all three: the medians that fill missing values, both
   standardisations and the intercept or baseline come from the combined rows.

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

A's value in each of the twelve rows is its row number, so you can see where a sampled row came
from. The fractions give pools of six, three and three rows. The same first test row has two
indices:

| Field | Counts within | Value |
| --- | --- | --- |
| `samples.evaluation.source_indices` | the test pool, `partition.evaluation` | 1 |
| `samples.evaluation_source_indices` | your original `Population` | 8 |

Use the second to link sampled rows to your own records; training and validation have the same
two fields.

- **Split by person or cluster when one contributes several rows.** `partition_population` splits
  individual rows, so related rows from the same person or cluster can end up in different
  pools, much as copies of a row can when you split after resampling. Build the
  `PopulationPartition` from your own row indices.
- **Fit your model's preprocessing on the training sample only**, as on real data; the pooled
  scaling of step 4 belongs to the simulation.
- **Targets hold on the combined rows.** With `probability=0.15`, event probabilities average 15%
  across all three samples, and one sample's fraction can differ.
- **The test rows are part of the generating model**: drawing different test rows can change the
  training rows' generating scores and outcome probabilities.
- **A randomly drawn sentinel that is constant or entirely missing in the training pool** raises
  `ValueError` because the feature cannot be standardised.

To compare a method across populations, such as two cohorts, call `generate` once per population
with the same effect. Each call standardises the score and calibrates the baseline on its own
rows, so the effect is per standard deviation of that population's score; with `target_auc`, each
call also solves its own effect ([AUC calibration](calibration.md#what-is-solved)). To apply one
fixed generating model to several populations, compute the score and draw the outcomes in your
own code.

## Signal construction

Every effect you set, such as an odds ratio, is per standard deviation of the generating score
(`truth.signal` in code), the number computed for each row from the generating features. The
score is built so that this unit means the same whatever the features' units and signal
structure. For the built-in signals PyPlasmode:

1. **fills missing values** of each generating feature with its median, so the score is defined on
   every row (`X` keeps its missing values);
2. **standardises each generating feature** to mean zero and population standard deviation one,
   so a weight measures a feature's contribution in standard deviations, not in its own units;
3. **combines the features** as the signal structure specifies (table below);
4. **standardises the combined score** to mean zero and population standard deviation one, so an
   odds ratio of 2 doubles the odds of the event for each one-standard-deviation increase in the
   score. A null score stays zero.

Because step 4 rescales the score, multiplying all weights by the same positive factor leaves
the score unchanged. Weights (4, −2) give the same score as (2, −1). If you name a generating
feature that is constant or entirely missing, generation raises `ValueError` because the feature
cannot be standardised. Automatic sparse selection skips such features.

In the table, a *sentinel* is the single generating feature chosen from a group of correlated
features, and a *module* is a named `FeatureSet` whose members all contribute ([Generating
signals](truths.md) explains when to choose each).

| Signal structure | How to construct the score | What it tests |
| --- | --- | --- |
| Sparse | Add the weighted values of selected features. | Recovery of individual generating features |
| Correlated, sentinels | Add the weighted values of one generating feature per group. | Recovery of generating features and their correlates |
| Distributed | Add weighted values from all module members. | Recovery of a shared signal |
| Correlated, group means | Average standardised values within each group. Apply group weights and sum the means directly. | Recovery of group-level information |
| Interaction, product | Multiply two standardised values. | Dependence on their combination |
| Interaction, joint threshold | Give one if both standardised values are strictly greater than `threshold` (default `0.0`), otherwise zero. | A relationship requiring both conditions |

`SparseTruth(count=k)` visits the features in an order set by the seed and accepts one only if its
absolute Pearson correlation with every feature already accepted, on the median-filled values, is at
most `max_absolute_correlation` (default 0.7). The limit stops two generating features from being
near-copies; otherwise a model that selected one would be scored as missing the other although the
two carry the same information. The search does not backtrack, so it can stop short of k even when k
such features exist, and generation then raises `ValueError`; ask for fewer features, raise the
limit or name the features yourself.

## Numerical calibration and random variation

Two checks answer the question "Does this dataset represent the scenario you specified?"

1. **Before drawing**, calibration sets the model's intercept or baseline so that the *expected*
   outcome over the sampled rows equals your target, to numerical precision. If the model
   ([Calibration equations](outcomes.md#calibration-equations)) cannot meet the target with the
   effect you set, generation raises `CalibrationError`. For example, a risk ratio of 3 with
   `probability=0.4` pushes the probabilities of the highest-scoring rows above one when the score
   is normally distributed; lower the risk ratio, or use an odds ratio, whose probabilities cannot
   leave the range 0 to 1.
2. **After drawing**, each observed summary is compared with its target, and generation raises
   `RealizationError` if they differ by more than about six standard errors.

With `probability=0.15`, the rows' event probabilities average 15%, and one drawn dataset might
contain 14.6% events. The expected value already equals the target. A `RealizationError` here
means the event fraction in this draw fell outside the tolerance, which is rare under the
specified model. With `p` the target probability and `n` the number of rows, the tolerances are:

- **Probabilities** (the binary event fraction, each ordered category's fraction, and survival
  event and censoring risks at each specified time): the larger of `6*sqrt(p*(1-p)/n)` and `6/n`.
  The second term allows a difference of six events when events are so rare that the first term
  would be smaller. Survival event risks are checked on the drawn event times before censoring, so
  heavy censoring does not by itself cause a failure.
- **Continuous outcomes**: `6*sd/sqrt(n)` for the mean and `6*sd/sqrt(2*(n-1))` for the standard
  deviation, with `sd` the target standard deviation.
- **Counts**: six times the standard deviation of the total count, divided by total exposure.

When you repeat a simulation, catch `RealizationError` and record the number of failures. If you
drop failed draws without reporting them, the study looks as if every draw succeeded.

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

The loop draws three 12-row datasets with a null signal and a 50% event probability. With 12
rows the tolerance is 0.87, which is wider than any possible distance from 50%, so these draws
cannot fail the event fraction check. The loop shows how to report failures when they can occur.
`summarize_replicates` reports the failure count alongside the mean of the successful draws, so
a reader of your results sees both.

## Discrimination and repeated selections

How close is the AUC from one dataset to `target_auc`? For binary and time-to-event outcomes,
calibration makes the expected AUC of the true risks, `outcome.calibration.oracle_auc`, equal
the target. The API calls this the *oracle* AUC ([AUC calibration](calibration.md)). The AUC
measured against the drawn outcomes can differ from that expectation, especially with few
events. When the effect raises risk, as it does here, the generating score gives the same
ranking as the true risks. If the effect lowers risk, multiply the generating score by the
negative `calibration["signal_coefficient"]` before measuring AUC.
then negative.

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

The smaller draw, with 18 events, is further from the target. Compare a model's AUC with the
realised AUC of the generating score in the same dataset. Both AUCs are measured against the
same drawn outcomes, whereas the target is an expected AUC.

When you refit a model on resampled data, its selected features change from fit to fit.
`selection_stability` measures how much the selections agree and `consensus_ranking` combines
them into a single ranking. For time-to-event outcomes, `cumulative_dynamic_auc` measures AUC
while accounting for censoring. All three are in `pyplasmode.metrics` ([Metrics](metrics.md)).
