# Calibrating to a target AUC

An odds ratio of 2 says how strongly the generating score acts on the outcome, but not how hard
the resulting prediction task is. The AUC (area under the ROC curve) of the generating score
says that directly: it is the probability that the score ranks a randomly chosen case above a
randomly chosen control. No model can expect to rank better than the true risks, so this AUC is
the ceiling for every method you test. Set `target_auc` on `BinaryOutcome` or
`TimeToEventOutcome`, and PyPlasmode solves for the odds ratio or hazard ratio per standard
deviation of the score that gives this expected AUC, while still matching your event
probability or event risks.

```python
import numpy as np
import pyplasmode as ppm

signal = np.random.default_rng(12).normal(size=20_000)
binary = ppm.generate_outcome(
    ppm.BinaryOutcome(probability=0.1, target_auc=0.7),
    signal,
    np.random.default_rng(3),
)
odds_ratio = binary.calibration["odds_ratio"]
print(round(binary.calibration.oracle_auc, 3), round(odds_ratio, 2))  # 0.7 2.15
```

`generate_outcome` draws an outcome from a score you supply, here 20,000 values from a standard
normal distribution, without sampling rows. An event probability of 10% and an AUC of 0.7 need
an odds ratio of 2.15. `oracle_auc` is the expected AUC the solved model achieves (the API calls
the AUC of the true risks the *oracle* AUC).

For time-to-event outcomes the AUC is measured at one time, `auc_horizon`: cases are people
with an event by then, and controls are people still event-free after it. The horizon must be
strictly before the end of follow-up, `administrative_horizon`, because otherwise no one is
observed past it to serve as a control in
[`cumulative_dynamic_auc`](metrics.md#censored-discrimination). The defaults follow from your
event-risk times:

- The end of follow-up defaults to the last event-risk time.
- The AUC horizon defaults to the last event-risk time strictly before the end of follow-up, or
  to half the end of follow-up if there is none.

So with a single ten-year risk, follow-up ends at ten years and the AUC is calibrated at five
years. To calibrate it at ten years, add a risk at a later time, such as eleven years: follow-up
then ends at eleven years and the AUC horizon defaults to ten. Without the later risk, setting
`administrative_horizon` to eleven years alone would leave no events after year ten, because
the event rate is zero after the last risk time.

```python
print(ppm.TimeToEventOutcome({ppm.years(10): 0.005}, target_auc=0.72).auc_horizon.days)  # 1826.25

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
hazard_ratio = survival.calibration["hazard_ratio"]
print(round(survival.calibration.oracle_auc, 3), round(hazard_ratio, 2))  # 0.72 2.29
```

The first line confirms the default horizon of five years (1,826.25 days). The second outcome is
a rare event, with risks of 0.5% by ten years and 0.6% by eleven, and it sets the AUC horizon to
ten years explicitly. A hazard ratio of 2.29 gives an expected ten-year AUC of 0.72. The target
and the solved hazard ratio use event risk before censoring, so censoring does not change them.
Censoring does reduce the number of observed events, which makes an AUC measured on the drawn
data less precise.

In one finite dataset, a fitted model's AUC can come out above or below the AUC of the
generating score measured on the same rows, because both are estimates with sampling error.
This is chance, not a sign that the model beats the true risks.

## What is solved

Knowing what the solver matches tells you what the target means and why the AUC of a drawn
dataset differs from it. PyPlasmode standardises the generating score on the supplied rows and
searches over the log odds ratio or log hazard ratio. At each trial value it solves for the
binary intercept, or the survival baseline hazards, that match your event probability or
event risks. For an AUC horizon between two risk times, it computes the baseline cumulative
hazard at the horizon from the constant rate in that interval.

The expected AUC is computed from the rows' event probabilities, with no outcomes drawn. Let
`p_i` be row `i`'s event probability, or, for survival, its risk by `auc_horizon` before
censoring. Each ordered pair of rows `(i, j)` is weighted by `p_i * (1 - p_j)`, the chance that
`i` is a case and `j` a control. The expected AUC is the share of the total pair weight for which
`p_i > p_j`, with half credit when they are equal. The two rows are drawn independently, so a
row is also paired with itself, as both case and control, and gets half credit.

The solver finds the log ratio at which the expected AUC equals `calibration.target_auc`. The
solved effect depends only on the score and the specification, so the same inputs always give
the same ratio. The AUC of a drawn dataset varies around the target, because it depends on
which rows have events and which events censoring hides.

The calibration result holds the solved values:

- `calibration["odds_ratio"]` or `calibration["hazard_ratio"]` is the solved ratio.
- `calibration["signal_coefficient"]` is its logarithm.
- `calibration.parameter_dict` returns all fitted parameters as a new dictionary, which you can
  change without changing the result.

`generate_partitioned` solves one effect on the training, validation and test rows together, so
all three samples share it. Separate calls to `generate` with `target_auc` each solve their own
effect, so the effect changes slightly from dataset to dataset, for example across sample sizes.
To hold the effect fixed, solve it once and pass the ratio as `odds_ratio` or `hazard_ratio` to
later calls. Each call still recalibrates the intercept or baseline hazards, so every dataset
keeps your event probability or risks.

## Invalid and unattainable targets

Each outcome takes exactly one effect setting, because each setting determines the effect on
its own. For a binary outcome set one of `odds_ratio`, `risk_ratio`, `risk_difference` or
`target_auc`; for a survival outcome set one of `hazard_ratio` or `target_auc`. Otherwise the
constructor raises `OutcomeSpecificationError`. It does the same for a target outside the range
from 0.5 (included) to 1 (excluded). An AUC of 0.5 is chance, and a score that ranks worse than
chance has a reversed sign, which the solver does not search. An AUC of 1 would need an infinite
ratio.

A survival AUC needs both cases and controls by the horizon. With `target_auc`, the constructor
therefore checks the event risk at the first risk time at or after `auc_horizon` (the last risk
time if the horizon is later) and raises `OutcomeSpecificationError` if it is 0, because that
leaves no cases. A risk of 1 is invalid at every risk time, because it leaves no controls.

Some valid targets cannot be reached with your score. Generation then raises `CalibrationError`:

- **A constant score** ranks no row above another, so it can reach only 0.5. A target of 0.5
  sets the ratio to 1 for any score.
- **Tied scores** limit the AUC, because a tied pair gets half credit whatever the ratio. A
  score built from one binary feature, for example, has only two values and cannot reach a high
  target.
- **The search range** covers log ratios from 0 to 8, ratios from 1 to about 2,981 per standard
  deviation. A target that needs more is refused rather than clipped.

To resolve the error, check the score for ties or lack of variation, or lower `target_auc`.

For survival, each row's hazard is the baseline hazard multiplied by `exp(beta * z)`, with
`beta` the log hazard ratio and `z` the standardised score. PyPlasmode limits `beta * z` to
the range −700 to 700 before exponentiating, so that the multiplier stays finite. At that
limit the event probabilities have already reached 0 or 1, so the limit does not change the
result.

## Validation

The [test suite](validation.md#test-suite) checks that the solved expected AUC equals the target
within `1e-7`. It also draws outcomes and measures the AUC they realise:

- **Binary:** 80,000 rows for each combination of event probability 0.02, 0.2 and 0.5 and
  target 0.5, 0.6, 0.75 and 0.9. The AUC of the true risks against the drawn outcomes is within
  0.025 of the target.
- **Survival:** 120,000 rows for each combination of ten-year event risk 0.005, 0.05 and 0.2,
  censoring risk 0, 0.05 and 0.3, and target 0.5, 0.61, 0.72 and 0.79. The AUC measured by
  `cumulative_dynamic_auc` is within 0.035 of the target.

These samples are large. In a smaller sample the realised AUC varies more, as
[Methods](methods.md#discrimination-and-repeated-selections) shows for 200 rows.
