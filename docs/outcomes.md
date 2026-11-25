# Outcomes in observable units

Specify outcomes by probabilities, means, rates or event risks. PyPlasmode calibrates
distribution parameters and stores them in `CalibrationResult`. Effect sizes refer to one
standard deviation of the generating score, the number calculated per row from the generating
features.

## Shared example population

The examples use four synthetic features, with A and B as the generating features. Run the
setup once, then run the outcome examples.

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

`BinaryOutcome` accepts a marginal positive probability and one odds ratio, risk ratio or risk
difference, which use logistic, log-risk and identity-risk links, respectively. The resulting
conditional probabilities must lie in `[0, 1]`; a risk ratio or risk difference fails
otherwise. Alternatively, `target_auc` solves the odds
ratio from a target AUC (area under the ROC curve) of the generating score; see the
[calibration guide](calibration.md).

```python
binary = simulate(ppm.BinaryOutcome(probability=0.15, odds_ratio=2.0))
print(binary.values.shape, round(float(binary.values.mean()), 3))
# (10000,) 0.152: one label per row; event fraction against a target of 0.15
```

Event odds double for a one-standard-deviation increase in the generating score.

## Continuous

`ContinuousOutcome` accepts a mean, standard deviation and either an effect in outcome units or
a target variance explained. Gaussian residual variation fills the remaining variance.
At least two rows are required. Use `effect=0` or `variance_explained=0` with a null signal.

```python
continuous = simulate(ppm.ContinuousOutcome(mean=100, standard_deviation=15, effect=5))
print(round(float(continuous.values.mean()), 2), round(float(continuous.values.std()), 2))
# 99.92 14.83: mean and standard deviation against targets of 100 and 15
```

A one-standard-deviation increase in the generating score raises the conditional mean by five
units of the measurement.

## Counts

`CountOutcome` generates integer counts such as hospital admissions over a
period of follow-up. Set `rate` to the expected total count divided by total
exposure. A `rate_ratio` of 1.5 raises the expected rate by 50% for each
one-standard-deviation increase in the generating score.

Set `exposure` to a finite positive amount shared by all rows (default `1.0`)
or supply one such amount per generated row. `counts.exposure` gives the
amount used for each row. Choose exposure units to match your rate, such as
years for admissions per year.

At a given score and exposure, `variance_to_mean` sets the variance divided by
the expected count. The default of 1 gives Poisson counts. A value above 1
gives negative-binomial (NB1) counts. With expected count 4 and
`variance_to_mean=2`, the variance is 8. The pooled variance also reflects
differences in expected counts across rows.

```python
counts = simulate(ppm.CountOutcome(rate=3, rate_ratio=1.5, variance_to_mean=2))
print(counts.values.dtype, round(float(counts.values.mean()), 3))
# int64 2.975: mean count per exposure unit against a target of 3

count_spec = ppm.CountOutcome(rate=3, rate_ratio=1.5, exposure=(1.0, 2.0))
varying_counts = ppm.generate_outcome(count_spec, np.array([-1.0, 1.0]), np.random.default_rng(7))
print(varying_counts.exposure, varying_counts.values)  # [1. 2.] [3 5]
```

The second draw assigns one year to the first row and two to the second.

## Ordered categories

`OrdinalOutcome` accepts ordered labels, their marginal probabilities and a common
odds ratio; proportional-odds thresholds are calibrated to those probabilities.
With a common odds ratio above one, a higher generating score shifts probability towards the
higher categories, with the same effect at every category boundary.

```python
ordinal = simulate(
    ppm.OrdinalOutcome(
        categories=("low", "middle", "high"),
        probabilities=(0.25, 0.50, 0.25),
        common_odds_ratio=1.6,
    )
)
print(ordinal.categories, np.bincount(ordinal.codes) / len(ordinal.codes))
# ('low', 'middle', 'high') [0.2524 0.4973 0.2503]: against targets of 0.25, 0.50, 0.25
```

The output stores category names and integer codes 0, 1 and 2.

## Time to event

Specify event risks at times relevant to your application. For example, risks of 10% at five years and 25% at ten years allow the
baseline event rate to change after year five. PyPlasmode uses a constant rate within each
interval, giving a piecewise-exponential model that matches these cumulative risks.

`TimeToEventOutcome` accepts cumulative event risks at one or more `days(...)` or `years(...)`
horizons, either a hazard ratio or `target_auc`, and optional cumulative censoring risks at
the same horizons. Censoring is independent. Administrative follow-up ends at the last
risk point unless a later `administrative_horizon` is supplied; no events occur after the
last risk point. See the [calibration guide](calibration.md) for `target_auc` and `auc_horizon`. Targets describe latent event risk before censoring.

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

`time` contains follow-up in days and `event` indicates whether the event was observed. The
observed event fraction is 0.235 (25% target latent risk at ten years). Censoring before an
event excludes it from the observed count. Fit survival models to
`time` and `event`; the returned latent event times are available for simulation checks.

## Calibration equations

In these equations, `z` is the standardised generating score. Calibration
chooses `alpha` (the intercept) where it appears.

- **Binary.** The row probability is `expit(alpha + log(odds_ratio)*z)` for an
  odds ratio, with `expit` the inverse logit. A risk ratio uses
  `exp(alpha + log(risk_ratio)*z)`. A risk difference uses
  `alpha + risk_difference*z`. The mean row probability equals `probability`.
- **Continuous.** Draw `mean + effect*z + epsilon`, where `epsilon` is
  independent zero-mean Gaussian noise with variance
  `standard_deviation**2 - effect**2`. Set `variance_explained` to get
  `effect = standard_deviation*sqrt(variance_explained)` or supply a signed
  `effect` directly.
- **Counts.** The expected row count is `mu = exposure*exp(alpha + log(rate_ratio)*z)`.
  Calibration sets `sum(mu)/sum(exposure) = rate`. A `CountOutcome(exposure=...)`
  vector describes generated rows, so resampling `X` leaves it in place.
  Supply `sample_size` entries for `generate` or the combined count in
  training, validation and evaluation order for `generate_partitioned`.
- **Ordered categories.** The probability of category `c` or lower is
  `expit(threshold_c - log(common_odds_ratio)*z)`. Calibration chooses
  `threshold_c` so its average equals the requested cumulative probability
  through `c`.
- **Time to event.** The event-free probability at time `t` is
  `S(t|z) = exp(-H0(t)*exp(log(hazard_ratio)*z))`. The baseline cumulative
  hazard `H0(t)` solves `mean(S(t|z)) = 1 - event_risk(t)` at each supplied
  risk point. Here `event_risk(t)` is the target before censoring.
  Linear interpolation of `H0(t)` gives a constant baseline rate within
  each interval. Observed time is the earliest of the drawn event time,
  independent censoring time and administrative cutoff.
