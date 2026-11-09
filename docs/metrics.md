# Metrics

Import the new evaluation functions from `pyplasmode.metrics`. They accept arrays
and nominations; they do not fit models. Align features and participants before
calling an array-based metric.

## Selection stability

Create one `FractionalNomination` per refit with `nomination_from_scores`, then
call `selection_stability`. Zero importance is excluded, so achieved budgets can
differ. Positive scores tied at the cutoff share the remaining slots uniformly.

```python
import pyplasmode as ppm
from pyplasmode.metrics import selection_stability, consensus_ranking, rank_biased_overlap

features = ("A", "B", "C", "D")
scores = [[4, 3, 0, 0], [4, 0, 0, 0], [4, 0, 3, 0]]
nominations = [ppm.nomination_from_scores(features, row, depth=2) for row in scores]
stability = selection_stability(nominations)
assert stability.membership_budgets == (2.0, 1.0, 2.0)
```

The definition is Nogueira et al.'s chance-corrected variance measure. For a
refit-by-feature selection probability matrix Q, the numerator is the mean across
features of `var(Q, ddof=1) + mean(Q*(1-Q))`. The denominator is `q*(1-q)`, where
q is the mean membership over all refits and features. Stability is one minus
their ratio. The additional term integrates independent tie resolutions; a
repeated completely uninformative tie has stability zero. The binary case is
Definition 4 of [Nogueira et al.](https://jmlr.org/papers/v18/17-514.html).
Negative estimates are retained. All-empty or all-full selections return
`value=None` because the chance denominator is zero. Unequal achieved budgets
cap maximum stability below one, even when the smaller selections are nested.
Selection stability summarises agreement across the supplied refits without a
confidence interval.

## Consensus and rank-biased overlap

`consensus_ranking(features, scores, depth=...)` orders features by mean fractional
inclusion in the leading list, then median rank among refits with positive
membership. Never-selected features receive median rank `float("inf")`. Equal evidence
retains an equal comparison rank. The `ranking` field uses feature names to order
storage within ties; use `ranks` for numerical comparisons.

```python
consensus = consensus_ranking(features, scores, depth=2)
assert consensus.ranking[0] == "A"
assert rank_biased_overlap([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
```

`rank_biased_overlap(left_ranks, right_ranks, persistence=0.9)` requires complete,
equally sized feature universes in the same array order. Lower ranks are better;
equal values indicate ties. **Convert importance scores, where higher is better,
with `scipy.stats.rankdata(-scores, method="average")` before passing them here.**

The default `tie_method="webber"` uses the published whole-tie convention:
include every item whose tie group starts at or before depth d, then compute
`2*len(intersection)/(len(left_prefix)+len(right_prefix))`. Identical tied lists
have agreement one. See Webber et al. (2010), section 4.3 (section 4.6, Eq. 28 in
[the author's version](https://www.codalism.com/research/papers/wmz10_tois.pdf)).
The prefix at depth d receives weight `(1-p)*p**(d-1)`; the complete-list tail
receives `p**(n-1)`.

Choose `tie_method="independent"` to average RBO over independent uniform
tie-breaking permutations instead. Its prefix agreement is the dot product of
fractional prefix memberships divided by d. A fully tied list has self-agreement
below one under this convention. The two definitions answer different questions.
The original `rank_stability` remains useful for pairwise Jaccard, Spearman and
untied RBO.

## Censored discrimination

`cumulative_dynamic_auc(training_time, training_event, time, event, risk,
horizon=..., tied_tolerance=0.0)` uses the training sample to estimate the
censoring distribution by reverse Kaplan–Meier. Cases have observed events at
or before the horizon. Controls have observed times strictly beyond it.
Censoring at or before the horizon does not create a control. Case weights are
`1/G(event_time)`; the controls' common horizon weight cancels. With event/censor
ties in the reference sample, events leave the risk set before censoring.
The default uses exact score ties with half credit. An explicit positive
`tied_tolerance` uses adjacent-score blocks, including transitive chains, to
reproduce the historical scikit-survival convention. Set `tied_tolerance=1e-8`
to match scikit-survival's default `tied_tol=1e-8`. This can merge dense
probabilities into large blocks; use it only when that convention is intended.

```python
from pyplasmode.metrics import cumulative_dynamic_auc

auc = cumulative_dynamic_auc(
    [1, 2, 4, 5],
    [1, 0, 1, 0],
    [1, 2, 4, 5],
    [1, 0, 1, 0],
    [0.9, 0.2, 0.1, 0.0],
    horizon=3,
)
assert auc == 1.0
```

Use one time unit throughout. To evaluate ten years in days, use
`horizon=ppm.years(10).days`. Both reference and evaluation follow-up must extend
beyond the horizon. To evaluate exactly at a generator's final event-risk point, set a later
`administrative_horizon`, or add a later event-risk point. The default generator
AUROC horizon is strictly before the administrative cutoff. With a landmark, subtract
the landmark from both follow-up and the absolute horizon.

The function raises `ValueError` for missing cases or controls, invalid arrays,
a horizon outside reference follow-up, or zero censoring survival. It assumes
independent censoring. It does not estimate competing-risk cumulative incidence
or support delayed entry directly. Separate training/evaluation arrays make the
censoring reference explicit; development data can serve as the reference if
that is the study's declared definition. See the
[scikit-survival definition](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.cumulative_dynamic_auc.html).

## Reproducing earlier values

Use `rank_biased_overlap(..., tie_method="independent")` for independent uniform
resolution of ranking ties, and `cumulative_dynamic_auc(..., tied_tolerance=1e-8)`
for scikit-survival's default score-tie tolerance. These options select distinct
definitions; keep them explicit when reproducing earlier analyses.

## Paired participant bootstrap and discrimination gap

The participant bootstrap uses one resampled index vector for both methods. A
callback returns their two statistics, or `None` if either is undefined. The
result is **right minus left**, with a percentile interval. Defaults are 1,000
attempts, 95% confidence and seed 0. Failures remain explicit in
`differences` and `failed_resamples`; at least two draws must succeed. Do not
catch arbitrary errors as undefined draws. The interval with failures is
conditional on success. A large failure count can make the interval misleading
in low-event samples, because it excludes zero-event resamples. Supply a row
count or a sequence of unique participant IDs. Models remain fixed; the callback controls whether any
evaluation nuisance parameters are re-estimated.

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from pyplasmode.metrics import paired_bootstrap_difference, discrimination_gap

labels = np.array([0, 0, 1, 1, 0, 1])
left = np.array([0.1, 0.3, 0.8, 0.6, 0.2, 0.9])
right = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])


def statistic(indices):
    if np.unique(labels[indices]).size < 2:
        return None
    return (
        roc_auc_score(labels[indices], left[indices]),
        roc_auc_score(labels[indices], right[indices]),
    )


comparison = paired_bootstrap_difference(
    len(labels),
    statistic,
    resamples=100,
    seed=7,
)
oracle_risk = np.array([0.1, 0.2, 0.4, 0.8, 0.5, 0.9])  # from the generating mechanism
gap = discrimination_gap(oracle_risk, left, metric=lambda scores: roc_auc_score(labels, scores))
assert np.isclose(gap.gap, -1 / 9)
```

Use `compare_paired_replicates` for independent simulation worlds. Its Monte
Carlo t interval answers a different question from a participant bootstrap.
`discrimination_gap` reports **oracle minus fitted** AUROC using one supplied
metric and shared outcomes, censoring reference and evaluation sample. Negative
sample gaps are possible. The gap measures how much discrimination the fitted
predictor loses relative to the true generating score on the evaluation sample.
`learnability_gap` remains an alias. To bootstrap this gap, return **fitted,
oracle** from the callback, so right-minus-left equals oracle-minus-fitted.
For a protective effect use `calibration["signal_coefficient"] * truth.signal`
as the oracle score, so higher scores still mean greater event risk.
