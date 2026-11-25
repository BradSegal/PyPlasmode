# Calibrating to a target AUC

Use `target_auc` when simulation difficulty is easier to specify than an odds or hazard ratio.
The target is the AUC (area under the ROC curve) of the generating score, the discrimination
attainable when the source of risk is known, reported as `oracle_auc`. A model fitted on finite
training data will typically have lower held-out AUC. The generator solves for a non-negative
coefficient for the score, giving a ratio of at least one per standard deviation of the
generating score while maintaining the requested marginal probability or event risks.

```python
import numpy as np
import pyplasmode as ppm

signal = np.random.default_rng(12).normal(size=20_000)
binary = ppm.generate_outcome(
    ppm.BinaryOutcome(probability=0.1, target_auc=0.7),
    signal,
    np.random.default_rng(3),
)
assert abs(binary.calibration.oracle_auc - 0.7) < 1e-7
odds_ratio = binary.calibration["odds_ratio"]
```

For survival, set `auc_horizon` strictly before the administrative cutoff.
It defaults to the last event-risk point before the cutoff, or half the cutoff
if no event-risk point precedes it. Interior horizons interpolate the cumulative
baseline hazard under the specified piecewise-constant rates. To target ten
years when the only risk point is at ten years, set
`administrative_horizon=ppm.years(11)`. Survivors then remain under observation without additional events or censoring
between the final risk point and that later cutoff. The example below instead
includes year 11 as a risk point, which also specifies additional
event and censoring risk between years 10 and 11. Independent censoring changes
the precision of a censoring-aware estimate, but not the AUC of the generating
score or the solved hazard ratio.

```python
horizon = ppm.years(10)
survival = ppm.generate_outcome(
    ppm.TimeToEventOutcome(
        event_risks={horizon: 0.005, ppm.years(11): 0.006},
        target_auc=0.72,
        auc_horizon=horizon,
        censoring_risks={horizon: 0.05, ppm.years(11): 0.055},
    ),
    signal,
    np.random.default_rng(3),
)
assert abs(survival.calibration.oracle_auc - 0.72) < 1e-7
hazard_ratio = survival.calibration["hazard_ratio"]
```

## What is solved

The generating score is standardised on the supplied rows. Binary outcomes use logistic
probabilities; survival outcomes use `1-exp(-H0(t)*exp(beta*z))`. For each trial
coefficient, the intercept or baseline hazard is recalibrated to the requested
marginal risk. Brent root-finding then matches the expected AUC of the
generating score.

The expectation is over case/control pairs drawn independently from the empirical distribution
of the generating score. The same source row can be drawn twice, and ties receive half credit.
`CalibrationResult.oracle_auc` records the achieved expectation and `target_auc` the request.
AUC realised in one outcome sample varies with the seed, sample size, event count and censoring.
The solved effect is deterministic for a given generating score and specification. Rerunning
with the same seed repeats the outcomes.

`generate_partitioned` calibrates once on the combined partition covariates,
so its returned samples share one solved effect. Calling `generate` separately
on training and test cohorts solves `target_auc` independently on each cohort.
Use `generate_partitioned` for a shared signal, or extract the solved ratio
with `train.outcome.calibration["odds_ratio"]` and pass it explicitly for test
outcomes. Their marginal baselines are still recalibrated separately.

`CalibrationResult.parameter_dict` returns a fresh parameter mapping;
`calibration["signal_coefficient"]` looks up one value.

## Invalid and unattainable targets

Specify exactly one binary effect measure or `target_auc`, and exactly one
survival `hazard_ratio` or `target_auc`. A target must lie in `[0.5, 1)`; invalid
specifications raise `OutcomeSpecificationError`. A survival target requires an
event risk above zero and below one at the first risk point at or after its horizon.

A target of 0.5 solves ratio one, including for a constant generating score, which
cannot attain a higher target. Ties and the distribution of the score can also limit
attainable discrimination. The solver supports log ratios from zero to eight,
independently of the largest value of the score. Logistic probabilities use `expit`
and the intercept bracket expands to cover the supplied offsets. Survival
multipliers saturate at `exp(-700)` and `exp(700)` to keep extreme tail hazards
finite. No observations are dropped or winsorised. A request outside the
supported effect range or the attainable discrimination raises
`CalibrationError`; it is never silently clipped.

## Validation

The statistical suite checks the expected AUC to absolute tolerance `1e-7`.
Generated binary AUCs are checked within 0.025 across probabilities 0.02, 0.2
and 0.5 and targets 0.5, 0.6, 0.75 and 0.9, using 80,000 synthetic observations
per cell. Survival checks use 120,000 observations, event risks 0.005, 0.05 and
0.2, censoring risks 0, 0.05 and 0.3, and targets 0.5, 0.61, 0.72 and 0.79;
the realised IPCW tolerance is 0.035. These are reproducible simulation checks,
not a guarantee of this accuracy for a user's smaller sample.
