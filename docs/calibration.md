# Calibrating oracle AUROC

Use `target_auc` when the intended simulation difficulty is easier to specify
than an odds or hazard ratio. The generator solves a non-negative signal
coefficient, while preserving the requested marginal probability or event risks.
The resulting ratio is at least one per signal standard deviation.

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
when there is only one point. Interior horizons interpolate the cumulative
baseline hazard under the specified piecewise-constant rates. To target exactly
ten years with one risk point, set `administrative_horizon=ppm.years(11)`.
Survivors then remain under observation without additional events or censoring
between the final risk point and that later cutoff. Independent censoring changes the precision of a
censoring-aware estimate, but not the population oracle AUROC or solved hazard
ratio. All event and censoring risk points are still calibrated. The example
includes year 11 to retain follow-up beyond the ten-year AUROC horizon; unlike
setting only a later administrative cutoff, this also specifies additional
event and censoring risk between years 10 and 11.

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

The signal is standardised on the supplied rows. Binary outcomes use logistic
probabilities; survival outcomes use `1-exp(-H0(t)*exp(beta*z))`. For each trial
coefficient, the intercept or baseline hazard is recalibrated to the requested
marginal risk. Brent root-finding then matches the expected oracle AUROC.

The expectation integrates case/control pairs over the empirical signal
population, with half credit for ties. It uses independent draws from that
population, including draws of the same source row. It does not optimise a noisy
AUROC from one realised outcome sample. `CalibrationResult.oracle_auc` records
this achieved expectation; `target_auc` records the request. Realised AUROC will
vary with the seed, sample size, event count and censoring. The solved effect is
deterministic for a given signal and specification; repeating the generator
with the same seed also repeats the outcomes.

`generate_partitioned` calibrates once on the combined partition covariates,
so its returned samples share one solved effect. Calling `generate` separately
on training and test cohorts solves `target_auc` independently on each cohort.
Use `generate_partitioned` for a shared mechanism, or extract the solved ratio
with `train.outcome.calibration["odds_ratio"]` and pass it explicitly for test
outcomes. Their marginal baselines are still recalibrated separately.

`CalibrationResult.parameter_dict` returns a fresh parameter mapping;
`calibration["signal_coefficient"]` looks up one value. For protective effects
(`odds_ratio < 1` or `hazard_ratio < 1`), use the solved coefficient times the
standardised signal as the oracle risk score. Passing the unflipped signal
would reverse the score direction. `target_auc` always solves a non-negative
coefficient.

## Invalid and unattainable targets

Specify exactly one binary effect measure or `target_auc`, and exactly one
survival `hazard_ratio` or `target_auc`. A target must lie in `[0.5, 1)`; invalid
specifications raise `OutcomeSpecificationError`. A survival target requires
positive event risk below one at its evaluation horizon.

A target of 0.5 solves ratio one, including constant signals. A constant signal
cannot attain a higher target. Ties and the signal distribution can also limit
attainable discrimination. The solver supports log ratios from zero to eight,
independently of the largest signal value. Logistic probabilities use `expit`
and the intercept bracket expands to cover the supplied offsets. Survival
multipliers saturate at `exp(-700)` and `exp(700)` to keep extreme tail hazards
finite. No observations are dropped or winsorised. A request outside the
supported effect range or the signal's attainable discrimination raises
`CalibrationError`; it is never silently clipped. The target describes the
true generating predictor: a model fitted to finite training data will usually
have lower held-out AUROC.

## Validation

The statistical suite checks expected AUROC to absolute tolerance `1e-7`.
Generated binary AUROCs are checked within 0.025 across probabilities 0.02, 0.2
and 0.5 and targets 0.5, 0.6, 0.75 and 0.9, using 80,000 synthetic participants
per cell. Survival checks use 120,000 participants, event risks 0.005, 0.05 and
0.2, censoring risks 0, 0.05 and 0.3, and targets 0.5, 0.61, 0.72 and 0.79;
the realised IPCW tolerance is 0.035. These are reproducible simulation checks,
not a guarantee of this accuracy for a user's smaller sample.
