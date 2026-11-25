# Calibrating to a target AUC

Set `target_auc` on `BinaryOutcome` or `TimeToEventOutcome` to choose the
expected AUC (area under the ROC curve) of the generating score. PyPlasmode
solves for an odds ratio or hazard ratio per standard deviation of that score
while matching your marginal probability or event risks. The achieved expected
AUC is `outcome.calibration.oracle_auc`.

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

For survival, set `auc_horizon` strictly before `administrative_horizon` so
people observed past the horizon can serve as controls in
[`cumulative_dynamic_auc`](metrics.md#censored-discrimination). The cutoff
defaults to the last event-risk time. The AUC horizon defaults to the last
event-risk time strictly before the cutoff, or half the cutoff if there is none.

With only a ten-year event-risk point and the default cutoff, calibration
targets five-year AUC. To target ten years, set `auc_horizon=ppm.years(10)`
and `administrative_horizon=ppm.years(11)`. Hazards are zero after the last
risk point. Add a year-11 risk point to specify event and censoring risk
between years 10 and 11. Independent censoring affects the precision of the
observed AUC estimate. The AUC target and solved hazard ratio use event risk
before censoring.

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

On a finite held-out sample, fitted AUC can fall below or exceed the measured
AUC of the generating score.

## What is solved

PyPlasmode standardises the generating score on the supplied rows and tries
values of its coefficient. At each trial value it solves for the binary
intercept or survival baseline hazards that match the requested marginal risks.
For an AUC horizon between risk points, it interpolates the baseline cumulative
hazard using the constant rate in that interval.

Let `p_i` be row `i`'s binary event probability or its risk by `auc_horizon`
before censoring. Each ordered pair `(i, j)` has case–control weight
`p_i * (1 - p_j)`. Expected AUC is the fraction of total pair weight for which
`p_i > p_j`, giving half credit when they tie. The two rows are drawn
independently, so the same row can serve as both case and control.

The solver matches this expected AUC to `calibration.target_auc`. The solved
effect is deterministic for the supplied score and specification. AUC realised
after sampling varies with the draw, including which events are observed.

Use `calibration["odds_ratio"]` or `calibration["hazard_ratio"]` to retrieve the
solved effect. `calibration["signal_coefficient"]` gives its logarithm for
AUC-targeted outcomes. `calibration.parameter_dict` returns a fresh copy of all
fitted parameters.

`generate_partitioned` solves one effect on the combined sampled rows.
Separate calls to `generate` with `target_auc` each solve a new effect. To
reuse an effect, pass the solved ratio explicitly to the next call. The
intercept or baseline hazards are still recalibrated to that sample.

## Invalid and unattainable targets

Set exactly one of `odds_ratio`, `risk_ratio`, `risk_difference` or
`target_auc` for binary outcomes. Set exactly one of `hazard_ratio` or
`target_auc` for survival outcomes. A target outside `[0.5, 1)` raises
`OutcomeSpecificationError`.

For survival, the first event-risk point at or after `auc_horizon` must have
risk in `(0, 1)`. If the horizon is after the last risk point, the last point
is checked. AUC needs both cases and controls. A zero risk at the checked point
leaves no cases by the horizon, and risk one is invalid at any supplied point.

A target of 0.5 sets the ratio to one. A constant generating score cannot
reach a higher target. Ties also limit how many case–control pairs the score
can distinguish. AUC calibration searches log ratios from 0 to 8, equivalent
to a ratio from 1 to about 2,981 per standard deviation. If the target is
unattainable, `CalibrationError` is raised. Check the score for ties or a lack
of variation, or lower `target_auc`.

For survival, the hazard factor is `exp(beta * z)`, with `beta` the log hazard
ratio and `z` the standardised generating score. PyPlasmode bounds `beta * z`
to `[-700, 700]` before exponentiating.

## Validation

The statistical suite checks the expected AUC to absolute tolerance `1e-7`.
Generated binary AUCs are checked within 0.025 across probabilities 0.02, 0.2
and 0.5 and targets 0.5, 0.6, 0.75 and 0.9, using 80,000 synthetic observations
per cell. Survival checks use 120,000 observations, event risks 0.005, 0.05 and
0.2, censoring risks 0, 0.05 and 0.3, and targets 0.5, 0.61, 0.72 and 0.79;
the realised IPCW tolerance is 0.035. These are reproducible simulation checks,
not a guarantee of this accuracy for a user's smaller sample.
