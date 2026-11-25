# Calibrating to a target AUC

A target AUC answers the question "how hard is the prediction task you are simulating?". An odds
ratio of 2 says how strongly the generating score acts on the outcome, but not that. The AUC
(area under the ROC curve) of the true risks does: it is the probability that a randomly chosen
case has a higher risk than a randomly chosen control, with ties counted as half. No model can
expect to rank better than the true risks, so this AUC is the ceiling for every method you test.

Set `target_auc` on `BinaryOutcome` or `TimeToEventOutcome` in place of an odds or hazard ratio.
PyPlasmode then:

1. searches over the log odds ratio or log hazard ratio per standard deviation of the score;
2. at each trial value, solves for the intercept, or the baseline hazards, that match your event
   probability or event risks;
3. computes the expected AUC of the true risks, without drawing outcomes
   ([What is solved](#what-is-solved));
4. keeps the ratio that gives the target and draws the outcome with it.

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

`generate_outcome` draws an outcome from a score you supply, here 20,000 standard normal values,
without sampling rows. An odds ratio of 2.15 gives this score an expected AUC of the true risks
(`oracle_auc`, which the API calls the *oracle* AUC) of 0.7 at a 10% event probability.

For a time-to-event outcome the AUC is measured at one time, `auc_horizon`: cases have an event by
then, and controls are still event-free after it. The horizon must be strictly before the end of
follow-up, `administrative_horizon`, because otherwise no one is observed past it to serve as a
control in [`cumulative_dynamic_auc`](metrics.md#censored-discrimination). Follow-up ends by
default at the last event-risk time, and the horizon defaults to the last event-risk time strictly
before that, or to half the end of follow-up if there is none. With a single ten-year risk:

| Setting | End of follow-up | AUC horizon |
| --- | --- | --- |
| Defaults | 10 years | 5 years |
| `administrative_horizon=ppm.years(11)` | 11 years | 10 years |

To calibrate the AUC at ten years, set the later end of follow-up: people still event-free at ten
years are then followed to eleven and serve as controls. No events occur after year ten, because
the event rate is zero after the last risk time; add a later, higher risk only if you want them.

```python
print(ppm.TimeToEventOutcome({ppm.years(10): 0.005}, target_auc=0.72).auc_horizon.days)  # 1826.25

survival = ppm.generate_outcome(
    ppm.TimeToEventOutcome(
        event_risks={ppm.years(10): 0.005},
        target_auc=0.72,
        censoring_risks={ppm.years(10): 0.05},
        administrative_horizon=ppm.years(11),
    ),
    signal,
    np.random.default_rng(3),
)
hazard_ratio = survival.calibration["hazard_ratio"]
print(survival.calibration["auc_horizon_days"])  # 3652.5
print(round(survival.calibration.oracle_auc, 3), round(hazard_ratio, 2))  # 0.72 2.29
```

The target and the solved hazard ratio use event risk before censoring, so the 5% censoring does
not change them. It does reduce the number of observed events, which makes an AUC measured on the
drawn data less precise.

In one finite dataset, a fitted model's AUC can come out above or below the AUC of the generating
score on the same rows, because both are estimates with sampling error. This is chance, not a sign
that the model beats the true risks.

## What is solved

The quantity the solver matches tells you what the target means and why a drawn dataset's AUC
differs from it. The expected AUC is computed from the rows' event probabilities. Let `p_i` be row
`i`'s event probability, or, for survival, its risk by `auc_horizon` before censoring. Each ordered
pair of rows `(i, j)` is weighted by `p_i * (1 - p_j)`, the chance that `i` is a case and `j` a
control. The expected AUC is the share of the total pair weight for which `p_i > p_j`, with half
credit when they are equal. The two rows are drawn independently, so a row is also paired with
itself, as both case and control, and gets half credit.

The solved ratio, in `calibration["odds_ratio"]` or `calibration["hazard_ratio"]`, depends only on
the score and the specification, so the same inputs always give the same ratio. The AUC of a drawn
dataset varies around the target, because it depends on which rows have events and which events
censoring hides.

`generate_partitioned` solves one effect on the training, validation and test rows together, so
all three samples share it. Separate calls to `generate` each solve their own effect, which
therefore changes from dataset to dataset, for example across sample sizes. To hold it fixed,
solve it once and pass the ratio as `odds_ratio` or `hazard_ratio` to later calls; each call still
recalibrates the intercept or baseline hazards, so every dataset keeps your event probability or
risks.

## Invalid and unattainable targets

The constructor refuses an invalid target with `OutcomeSpecificationError`:

- **Anything but exactly one effect setting**, because each setting determines the effect on its
  own: one of `odds_ratio`, `risk_ratio`, `risk_difference` or `target_auc` for a binary outcome,
  one of `hazard_ratio` or `target_auc` for survival.
- **A target outside the range from 0.5 (included) to 1 (excluded).** An AUC of 0.5 is chance, and
  a score that ranks worse than chance has a reversed sign, which the solver does not search. An
  AUC of 1 needs every true risk to be exactly 0 or 1, which no finite ratio gives.
- **A survival target with no cases.** A survival AUC needs cases by the horizon, so a zero event
  risk at the first risk time at or after `auc_horizon` (the last risk time if the horizon is
  later) is refused. A risk of 1 is invalid at every risk time.

Generation raises `CalibrationError` for a valid target your score cannot reach:

- **A constant score** ties every pair, so it can reach only 0.5; a target of 0.5 sets the ratio
  to 1 for any score.
- **Tied scores** can cap the AUC, because a case and a control with the same risk get half
  credit whatever the ratio. How much depends on how the events can fall between the tied values:
  with a 10% event probability, a score from one binary feature present in half the rows cannot
  reach 0.8, but one present in a tenth of the rows can reach 0.99.
- **The search range** covers log ratios from 0 to 8, ratios from 1 to about 2,981 per standard
  deviation. A target that needs more is refused rather than clipped.

To resolve it, check the score for ties or lack of variation, or lower `target_auc`.

For survival, each row's log hazard multiplier, `beta * z`, is limited to −700 to 700 before
exponentiating, so that the multiplier stays finite and positive.

## Validation

The [test suite](validation.md#test-suite) checks that the solved expected AUC equals the target
within `1e-7`, over a grid of event risks and targets, with censoring also varied for survival.
It also draws outcomes: the AUC of the true risks is within 0.025 of the target for 80,000 binary
rows, and within 0.035 for 120,000 survival rows measured with `cumulative_dynamic_auc`. In a
smaller sample the realised AUC varies more, as
[Methods](methods.md#discrimination-and-repeated-selections) shows for 200 rows.
