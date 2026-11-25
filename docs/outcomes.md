# Outcomes in observable units

A simulated outcome should look like the outcome in your setting: a 15% event rate, a mean blood
pressure of 120, a 10% risk at five years. You therefore describe it in those observable units,
and PyPlasmode solves for the model parameters, such as an intercept, that produce them. Each
outcome takes two kinds of setting: targets for the outcome's level, and an effect for how
strongly the generating score acts on it, which a target AUC can replace for binary and
time-to-event outcomes. Every effect is per standard deviation of the
generating score, the number computed for each row from the generating features. The solved
parameters are stored with the outcome in `outcome.calibration`.

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

A risk ratio or risk difference can push the probability of the highest- or lowest-scoring rows
outside 0 to 1, and generation then raises `CalibrationError`; use a smaller effect or an odds
ratio, which keeps every probability in range whatever the score. To set how well the score
separates cases from controls instead, set `target_auc` and PyPlasmode solves for the odds ratio
([AUC calibration](calibration.md)).

```python
binary = simulate(ppm.BinaryOutcome(probability=0.15, odds_ratio=2.0))
print(binary.values.shape, round(float(binary.values.mean()), 3))
print(round(binary.calibration["intercept"], 2))
# (10000,) 0.152
# -1.89
```

The outcome holds one label per row, with an event fraction close to the 15% target. The
intercept is the log-odds value calibration found to make the row probabilities average 15%.

## Continuous

Use `ContinuousOutcome` for a measured quantity such as blood pressure. Set its `mean` and
`standard_deviation`, and one of two effects:

- `effect`, the change in the mean for each one-standard-deviation increase in the score, when
  you think in the outcome's units. Its size must be smaller than `standard_deviation`, so that
  some variance is left for the Gaussian noise that supplies the rest.
- `variance_explained`, the share of the outcome's variance due to the score (at least 0 and
  below 1), when you think in terms of how predictable the outcome is. It always gives a
  positive effect; use `effect` for a negative one.

With a null signal, set the effect to 0:
a constant score cannot explain any variance, and any other value raises
`OutcomeSpecificationError`. At least two rows are needed, because a
standard deviation cannot be checked on one.

```python
continuous = simulate(ppm.ContinuousOutcome(mean=100, standard_deviation=15, effect=5))
print(round(float(continuous.values.mean()), 2), round(float(continuous.values.std()), 2))
# 99.92 14.83
```

The sample's mean and standard deviation are close to their targets of 100 and 15. The
conditional mean rises by five units for each one-standard-deviation increase in the
score, so the score explains (5/15)² = 11% of the variance.

## Counts

Use `CountOutcome` for counts over a period of follow-up, such as hospital admissions. Set
`rate` to the expected total count divided by the total exposure, for example admissions per
person-year. A `rate_ratio` of 1.5 raises the expected rate by 50% for each one-standard-deviation
increase in the score.

`exposure` is each row's follow-up, in the units of the rate's denominator: years for a rate per
year. Give one value for all rows (default 1.0) or one per simulated row, not per source row:
`sample_size` values for `generate`, or one per row of the three samples, in training, validation
and test order, for `generate_partitioned`. `counts.exposure` returns the value used for each row.

`variance_to_mean` sets how variable the counts are among rows with the same score and exposure.
The default of 1 gives Poisson counts, whose variance equals their expected count. A value above
1 gives overdispersed, negative-binomial counts (the NB1 form), whose variance is that multiple of
the expected count. Real counts are often overdispersed, because people
with the same measurements still differ in ways the measurements miss.

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

The first draw gives integer counts with a mean close to the target of 3 per unit of exposure.
Their variance is larger than twice the mean, because `variance_to_mean` applies
among rows with the same score, and the score also makes expected counts differ between rows.
The second draw calls `generate_outcome`, which draws outcomes from a score you supply without
sampling rows, here two rows with scores −1 and 1 and exposures of one and two years.

## Ordered categories

Use `OrdinalOutcome` for ordered categories such as mild, moderate and severe disease. Set
`categories` from lowest to highest, `probabilities`, the share of rows in each in the same
order, and a `common_odds_ratio`.
PyPlasmode uses a proportional-odds model: at every boundary between categories, the odds of
being above it are multiplied by the same ratio for each one-standard-deviation increase in the
score. A ratio above one therefore moves rows with a higher score towards the higher
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

The outcome stores the category names and an integer code per row, 0 for the lowest category.
The shares of rows are close to their targets.

## Time to event

Use `TimeToEventOutcome` for the time until an event such as a diagnosis, when not every
person is followed until it happens. Set the cumulative event risk at one or more times, built
with `days(...)` or `years(...)`. Risks of 10% at five years and 25% at ten years, for example,
let the baseline event rate change after year five. PyPlasmode holds each row's event rate (its
hazard) constant between consecutive times and matches the average risk at each time (a
piecewise-exponential model).

Set the effect as a `hazard_ratio` per standard deviation of the score, or set `target_auc`
([AUC calibration](calibration.md)). Three further settings shape follow-up:

- **Risk targets exclude censoring.** Each event risk is the risk in the absence of censoring,
  so your targets do not depend on how much follow-up is lost.
- **Censoring is optional and independent.** `censoring_risks` gives the cumulative risk of
  losing a person to follow-up, at the same times as the event risks. Censoring does not depend
  on the score or on the event, but it hides later events more often, because they have longer
  to be censored.
- **Follow-up ends at the last risk time** unless you set a later `administrative_horizon`, which
  follows people still event-free for longer. The event and censoring rates are zero after the
  last risk time, so to have events after it, add a later risk time with a higher risk.

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

Fit survival models to `time`, the follow-up in days, and `event`, which is true when the event was
observed. The observed event fraction is below the 25% risk at ten years, because censoring ends
some people's follow-up before their event. The drawn times before censoring, `latent_event_time`
and `censoring_time`, are kept for checking the simulation; a time is infinite when it falls after
the last risk time.

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
