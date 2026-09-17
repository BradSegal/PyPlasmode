# Outcomes in observable units

Specify outcomes using probabilities, means, rates or event risks. PyPlasmode calibrates
the distribution parameters and records them in `CalibrationResult`. Effect sizes refer
to one standard deviation of the complete generating signal.

## Shared example population

The examples use a synthetic proteomic population and the same two contributing proteins.
Run this setup once, then each family below. In an application, `Population` accepts your
measured matrix and protein identifiers. All outputs contain one response per sampled row.

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
difference. These use logistic, log-risk and identity-risk links, respectively. The resulting
conditional probabilities must lie in `[0, 1]`.

```python
binary = simulate(ppm.BinaryOutcome(probability=0.15, odds_ratio=2.0))
print(binary.values.shape, round(float(binary.values.mean()), 3))
# (10000,) 0.152
```

This returns 10,000 zero/one labels with an event fraction near 0.15. The odds of an
event double for a one-standard-deviation increase in the generating score.

## Continuous

`ContinuousOutcome` accepts a mean, standard deviation and either an effect in outcome units or
a target variance explained. Gaussian residual variation fills the remaining variance.
At least two rows are required. Use `effect=0` or `variance_explained=0` with a null signal.

```python
continuous = simulate(ppm.ContinuousOutcome(mean=100, standard_deviation=15, effect=5))
print(round(float(continuous.values.mean()), 2), round(float(continuous.values.std()), 2))
# 99.92 14.83
```

The response has mean near 100 and standard deviation near 15, in the units of the
clinical measurement. A one-standard-deviation increase in the signal raises its
conditional mean by five units.

## Counts

Counts describe how many events occur during an exposure period. `CountOutcome` takes the
average rate per unit exposure and a rate ratio per signal standard deviation. For example,
`CountOutcome(rate=3, rate_ratio=1.5, variance_to_mean=2)` specifies three events per exposure
unit on average, a 50% higher conditional rate per signal standard deviation, and extra
variation beyond a Poisson model.

The variance-to-mean ratio controls variation among participants with the same signal and
exposure. If their expected count is 4, a ratio of 2 gives variance 8. A ratio of 1 gives the
Poisson model, whose variance equals its mean. Ratios above 1 use the NB1 negative-binomial
model: variance increases in direct proportion to the conditional mean. This differs from
the pooled variance across people, which also reflects differences in their expected counts.
For reproduction, the negative-binomial shape is `mu / (variance_to_mean - 1)` and its gamma
mixing scale is `variance_to_mean - 1`.

```python
counts = simulate(ppm.CountOutcome(rate=3, rate_ratio=1.5, variance_to_mean=2))
print(counts.values.dtype, round(float(counts.values.mean()), 3))
# int64 2.975
```

The result contains integer event counts, averaging approximately three per unit exposure.
For a hospital-admission application, an exposure unit could be one year. The rate rises
by 50% per signal standard deviation; the conditional variance is twice the mean.

## Ordered categories

`OrdinalOutcome` accepts ordered labels, their marginal probabilities and a common
odds ratio; proportional-odds thresholds are calibrated to those probabilities.
For example, categories `low`, `middle` and `high` can have probabilities 0.25, 0.50 and 0.25.
A positive signal shifts probability towards the higher categories; the common odds ratio
uses the same effect at both category boundaries.

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

The output stores category names and integer codes 0, 1 and 2. Category proportions
are close to the requested quarters/half/quarter, with larger signals favouring greater
severity.

## Time to event

Specify event risks at clinically meaningful times rather than choosing distribution
parameters. For example, risks of 10% at five years and 25% at ten years allow the baseline
event rate to change after year five. PyPlasmode uses a constant rate within each interval,
giving a piecewise-exponential model that matches those cumulative risks.

`TimeToEventOutcome` accepts cumulative event risks at one or more `days(...)` or `years(...)`
horizons, a hazard ratio and optional cumulative censoring risks at the same horizons. One
point implies a constant baseline hazard. Several points imply piecewise-constant interval
hazards. Censoring is independent and administrative follow-up ends at the last horizon.
Targets describe latent event risk before censoring; observed time and event state are returned
for modelling.

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

`time` contains follow-up in days and `event` indicates an observed diagnosis. The
observed event fraction is usually below the requested 25% ten-year latent event risk
because some participants are censored before their event. Fit survival models to
`time` and `event`; the returned latent event times are available for simulation checks.

## Calibration equations

For standardised signal `z`, binary odds-ratio generation uses
`P(Y=1|z) = expit(alpha + log(OR)*z)`, solving `mean(P)=probability` for `alpha`.
Risk-ratio generation uses `exp(alpha + log(RR)*z)`; risk-difference generation uses
`probability + RD*z`. The latter two fail if a conditional probability leaves `[0, 1]`.

Continuous generation uses `Y = mean + effect*z + epsilon`, with independent Gaussian
`epsilon` of variance `standard_deviation^2 - effect^2`. For requested `R2`,
`effect = standard_deviation*sqrt(R2)`. A signed effect can be supplied directly.

For counts, `mu_i = exposure_i*exp(alpha + log(rate_ratio)*z_i)` and
`sum(mu_i)/sum(exposure_i) = rate`. Exposures refer to generated rows in their output order;
an exposure vector is not resampled as if it were a source-population column.

Ordinal generation uses `P(Y <= c|z) = expit(threshold_c - log(common_odds_ratio)*z)`.
Each threshold matches its requested marginal cumulative category probability.

Survival generation uses `S(t|z) = exp(-H0(t)*exp(log(hazard_ratio)*z))`.
At each supplied horizon, solve `mean(S(t|z)) = 1 - event_risk(t)`; interpolate cumulative
hazard linearly between horizons, giving constant interval hazards. Event and independent
censoring thresholds are drawn from unit exponentials and inverted against those cumulative
hazards. Observed time is the minimum of event, censoring and administrative time.
Latent event risks are not observed event proportions after censoring.
