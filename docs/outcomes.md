# Outcomes in observable units

An outcome specification answers the question "What should the outcome look like, and how
strongly should the generating score act on it?" To match the outcome in your setting, describe
it in observable units, such as a 15% event rate or a 10% risk at five years. PyPlasmode solves
for the model parameters, such as an intercept, that produce the requested values.

Each class takes targets for the outcome's level and one effect, per standard deviation of the
generating score (the number computed for each row from the generating features); a target AUC
can be provided instead of the effect for binary and time-to-event outcomes. The solved
parameters are stored in `outcome.calibration`.

## Shared example population

The examples share one population of 10,000 rows and four independent features, with A and B as
the generating features. `simulate` draws all 10,000 rows once each with the same seed, so the
examples differ only in their outcome model. Run this block first.

```python
import numpy as np
import pyplasmode as ppm

population = ppm.Population(np.random.default_rng(7).normal(size=(10_000, 4)), ("A", "B", "C", "D"))
truth = ppm.SparseTruth(features=("A", "B"))


def simulate(specification):
    return ppm.generate(
        population,
        truth=truth,
        outcome=specification,
        sample_size=10_000,
        seed=42,
        sampling_method="without_replacement",
    ).outcome
```

## Binary

Use `BinaryOutcome` for a yes-or-no outcome such as a diagnosis. Set `probability`, the event
probability averaged over rows, and one effect on the scale your field reports:

- `odds_ratio` multiplies the odds of the event (a logistic model).
- `risk_ratio` multiplies the event probability (a log-risk model).
- `risk_difference` adds to the event probability (an identity-risk model).
- `target_auc` sets how well the score separates cases from controls, and PyPlasmode solves for
  the odds ratio ([AUC calibration](calibration.md)).

A risk ratio or risk difference may push the probability of the highest- or lowest-scoring rows
outside 0 to 1. Generation then raises `CalibrationError`. Use a smaller effect or an odds
ratio, which keeps all probabilities in range.

```python
binary = simulate(ppm.BinaryOutcome(probability=0.15, odds_ratio=2.0))
print(binary.values.shape, round(float(binary.values.mean()), 3))
print(round(binary.calibration["intercept"], 2))
# (10000,) 0.152
# -1.89
```

The event fraction, 0.152, is close to the 15% target. The intercept is the log-odds value that
makes the row probabilities average 15%.

## Continuous

Use `ContinuousOutcome` for a measured quantity such as blood pressure. Set `mean`,
`standard_deviation` and one effect: `effect`, the change in the mean per one-standard-deviation
increase in the score, in the outcome's units; or `variance_explained`, the share of the
outcome's variance due to the score. With `standard_deviation=15`, `effect=5` and
`variance_explained=1/9` give the same outcome, because (5/15)² = 1/9.

- **`effect` must be smaller than `standard_deviation` in size**, to leave variance for the
  Gaussian noise that supplies the rest.
- **`variance_explained` is at least 0 and below 1, and gives an effect of zero or more**; use
  `effect` for a negative one.
- **With a null signal the effect must be 0**, because a constant score cannot explain any
  variance; any other value raises `OutcomeSpecificationError`.
- **At least two rows are needed**, because a standard deviation cannot be checked on one.

```python
continuous = simulate(ppm.ContinuousOutcome(mean=100, standard_deviation=15, effect=5))
print(round(float(continuous.values.mean()), 2), round(float(continuous.values.std()), 2))
# 99.92 14.83
```

The sample mean and standard deviation are close to the targets of 100 and 15.

## Counts

Use `CountOutcome` for counts over a period of follow-up, such as hospital admissions. Set:

- `rate`, the expected total count divided by the total exposure, such as admissions per
  person-year;
- `rate_ratio`; 1.5 raises the expected rate by 50% per one-standard-deviation increase in the
  score;
- `exposure`, each row's follow-up in the rate's denominator (years for a rate per year): one
  value for all rows (default 1.0), or one per simulated row, not per source row (in training,
  validation and test order for `generate_partitioned`). `counts.exposure` returns the values
  used;
- `variance_to_mean`, how variable counts are among rows with the same score and exposure. The
  default of 1 gives Poisson counts, whose variance equals their expected count; a value above 1
  gives negative-binomial counts (the NB1 form) with variance that multiple of the expected
  count. Real counts are often overdispersed, because people with the same measurements still
  differ in ways the measurements miss.

```python
counts = simulate(ppm.CountOutcome(rate=3, rate_ratio=1.5, variance_to_mean=2))
print(
    counts.values.dtype, round(float(counts.values.mean()), 3), round(float(counts.values.var()), 1)
)
# int64 2.975 7.4

count_spec = ppm.CountOutcome(rate=3, rate_ratio=1.5, exposure=(1.0, 2.0))
varying_counts = ppm.generate_outcome(count_spec, np.array([-1.0, 1.0]), np.random.default_rng(7))
print(varying_counts.exposure, varying_counts.values)  # [1. 2.] [3 5]
```

The first draw's mean is close to the target of 3. The variance is more than twice the mean
because `variance_to_mean` applies to counts at the same score and exposure, and the score also
causes the expected counts to differ between rows. For the second draw, `generate_outcome` draws
from a score you provide without sampling rows. Here you supply two rows with scores of −1 and 1
and exposures of one and two years.

## Ordered categories

Use `OrdinalOutcome` for ordered categories such as mild, moderate and severe disease. Set
`categories` from lowest to highest, `probabilities`, the share of rows in each in the same
order, and a `common_odds_ratio`. In this proportional-odds model, the odds of being above every
boundary between categories are multiplied by the same ratio for each one-standard-deviation
increase in the score, so a ratio above one moves rows with a higher score towards the higher
categories. Calibration sets the boundaries so that the shares match your targets.

```python
ordinal = simulate(
    ppm.OrdinalOutcome(
        categories=("low", "middle", "high"),
        probabilities=(0.25, 0.50, 0.25),
        common_odds_ratio=1.6,
    )
)
print(ordinal.categories, np.bincount(ordinal.codes) / len(ordinal.codes))
# ('low', 'middle', 'high') [0.2524 0.4973 0.2503]
```

The outcome stores the category names and an integer code per row, with 0 being the lowest
category.

## Time to event

Use `TimeToEventOutcome` for the time until an event such as a diagnosis, when not everyone is
followed until it happens. Set the cumulative event risk at one or more times, built with
`days(...)` or `years(...)`, and a `hazard_ratio` per standard deviation of the score or a
`target_auc` ([AUC calibration](calibration.md)). Each row's event rate (its hazard) is constant
between consecutive risk times, and the average risk matches your target at each time (a
piecewise-exponential model): risks of 10% at five years and 25% at ten let the baseline rate
change after year five.

- **Risk targets exclude censoring**, so they do not depend on how much follow-up is lost.
- **Censoring is optional and independent.** `censoring_risks` gives the cumulative risk of loss
  to follow-up at the same times. It does not depend on the score or the event, but hides later
  events more often, because they have longer to be censored.
- **Follow-up ends at the last risk time** unless you set a later `administrative_horizon`. Event
  and censoring rates are zero after the last risk time, so for events after it, add a later risk
  time with a higher risk.

```python
survival = simulate(
    ppm.TimeToEventOutcome(
        event_risks={ppm.years(5): 0.10, ppm.years(10): 0.25},
        hazard_ratio=1.7,
        censoring_risks={ppm.years(5): 0.05, ppm.years(10): 0.10},
    )
)
print(survival.time.shape, survival.event.dtype)
print(round(float(survival.event.mean()), 3))
# (10000,) bool
# 0.235
```

Fit survival models to `time`, the follow-up in days, and `event`, true when the event was
observed. The event fraction, 0.235, is below the 25% risk at ten years, because censoring ends
some follow-up before the event. The drawn times before censoring, `latent_event_time` and
`censoring_time`, are kept for checking the simulation; a time is infinite when it falls after the
last risk time.

## Calibration equations

These are the models behind each outcome. Use them to work out what a setting implies, or to
reimplement a model in your own code. Throughout, `z` is the standardised generating score and
calibration chooses the intercept `alpha` where it appears.

- **Binary.** The row probability is `expit(alpha + log(odds_ratio)*z)` for an
  odds ratio, with `expit` the inverse logit. A risk ratio uses
  `exp(alpha + log(risk_ratio)*z)`. A risk difference uses
  `alpha + risk_difference*z`. The mean row probability equals `probability`.
- **Continuous.** Draw `mean + effect*z + epsilon`, where `epsilon` is
  independent zero-mean Gaussian noise with variance
  `standard_deviation**2 - effect**2`. Setting `variance_explained` gives
  `effect = standard_deviation*sqrt(variance_explained)`.
- **Counts.** The expected row count is `mu = exposure*exp(alpha + log(rate_ratio)*z)`.
  Calibration sets `sum(mu)/sum(exposure) = rate`. The conditional variance is
  `variance_to_mean*mu`.
- **Ordered categories.** The probability of category `c` or lower is
  `expit(threshold_c - log(common_odds_ratio)*z)`. Calibration chooses
  `threshold_c` so its average equals the requested cumulative probability
  through `c`.
- **Time to event.** The event-free probability at time `t` is
  `S(t|z) = exp(-H0(t)*exp(log(hazard_ratio)*z))`. The baseline cumulative
  hazard `H0(t)` solves `mean(S(t|z)) = 1 - event_risk(t)` at each supplied
  risk time, with `event_risk(t)` the target before censoring.
  `H0(t)` rises linearly from zero between risk times and stays flat after
  the last one, so the baseline rate is constant within each interval and zero
  afterwards. Censoring has its own cumulative hazard
  `-log(1 - censoring_risk(t))` at the same times, interpolated the same way. Observed time
  is the earliest of the drawn event time, the censoring time and the end of
  follow-up.
