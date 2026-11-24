# Metrics

`pyplasmode.metrics` measures agreement between repeated feature selections, AUC (area
under the ROC curve) under censoring, and differences between two methods on the same
observations. Its functions accept arrays and selections; they do not fit models. Align
features and observations before calling an array-based metric.

## Selection stability

`selection_stability` measures how consistently refits of a method select the same
features, corrected for chance. Create one selection per refit with
[`nomination_from_scores`](evaluation.md#rankings-and-scores), which returns a
`FractionalNomination`. Zero importance is excluded, so membership budgets can differ.

```python
import pyplasmode as ppm
from pyplasmode.metrics import selection_stability, consensus_ranking, rank_biased_overlap

features = ("A", "B", "C", "D")
scores = [[4, 3, 0, 0], [4, 0, 0, 0], [4, 0, 3, 0]]
nominations = [ppm.nomination_from_scores(features, row, depth=2) for row in scores]
stability = selection_stability(nominations)
assert stability.membership_budgets == (2.0, 1.0, 2.0)
```

The definition extends the chance-corrected variance measure of
[Nogueira et al.](https://jmlr.org/papers/v18/17-514.html), Definition 4, from binary
selections to ties. For a refit-by-feature selection probability matrix Q, the numerator is the mean across
features of `var(Q, ddof=1) + mean(Q*(1-Q))`. The denominator is `q*(1-q)`, where
q is the mean membership over all refits and features. Stability is one minus
their ratio. The additional term integrates independent tie resolutions; a
repeated completely uninformative tie has stability zero.
Negative estimates are retained. All-empty or all-full selections return
`value=None` because the chance denominator is zero. Unequal membership budgets
cap maximum stability below one, even when the smaller selections are nested.

## Consensus and rank-biased overlap

`consensus_ranking` combines the importance scores from several refits into one ranking;
`rank_biased_overlap` (RBO) compares two rankings that may contain ties.

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
The prefix at depth d receives weight `(1-p)*p**(d-1)`; the complete-list tail
receives `p**(n-1)`. For untied rankings,
[`rank_stability`](evaluation.md#ranking-stability) gives Jaccard, Spearman and RBO.

The default `tie_method="webber"` uses the published whole-tie convention:
include every item whose tie group starts at or before depth d, then compute
`2*len(intersection)/(len(left_prefix)+len(right_prefix))`. Identical tied lists
have agreement one. See Webber et al. (2010), Equation 28 in section 4.6 of
[the author's manuscript](https://www.codalism.com/research/papers/wmz10_tois.pdf).

Choose `tie_method="independent"` to average RBO over independent uniform
tie-breaking permutations instead. Its prefix agreement is the dot product of
fractional prefix memberships divided by d. A fully tied list has self-agreement
below one under this convention.

## Censored discrimination

`cumulative_dynamic_auc(training_time, training_event, time, event, risk,
horizon=..., tied_tolerance=0.0)` estimates AUC at one time horizon from censored
follow-up. It uses the training sample to estimate the
censoring distribution by reverse Kaplan–Meier. Cases have observed events at
or before the horizon. Controls have observed times strictly beyond it.
Censoring at or before the horizon does not create a control. Case weights are
`1/G(event_time)`; the controls' common horizon weight cancels. With event/censor
ties in the training sample, events leave the risk set before censoring.

The default uses exact score ties with half credit. An explicit positive
`tied_tolerance` uses adjacent-score blocks, including transitive chains, which is
the scikit-survival convention. Set `tied_tolerance=1e-8`
to match scikit-survival's default `tied_tol=1e-8`. This can merge dense
probabilities into large blocks; use it only when that convention is intended. It matches
how ties are scored; the two libraries still differ in the follow-up they require.

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
`horizon=ppm.years(10).days`. Both training and evaluation follow-up must extend
beyond the horizon; [calibration](calibration.md) shows how to generate follow-up
beyond a chosen horizon. With a landmark, subtract
the landmark from both follow-up and the absolute horizon.

The function raises `ValueError` for missing cases or controls, invalid arrays,
a horizon that is not positive and finite or that reaches the maximum training follow-up,
or zero censoring survival. It assumes
independent censoring and does not estimate competing-risk cumulative incidence
or support delayed entry. See the
[scikit-survival definition](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.cumulative_dynamic_auc.html).

## Paired bootstrap and discrimination gap

`paired_bootstrap_difference` gives an interval for the difference between two fixed
fitted models evaluated on the same observations. It uses one resampled index vector
for both methods. A
callback returns their two statistics, or `None` if either is undefined. The
result is **right minus left**, with a percentile interval. Defaults are 1,000
attempts, 95% confidence and seed 0. Failures remain explicit in
`differences` and `failed_resamples`; at least two draws must succeed. Do not
catch arbitrary errors as undefined draws. The interval with failures is
conditional on success. A large failure count can make the interval misleading
in low-event samples, because it excludes zero-event resamples. Pass a row
count or a sequence of unique IDs as `participants`. Models remain fixed; the callback controls whether any
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
oracle_risk = np.array([0.1, 0.2, 0.4, 0.8, 0.5, 0.9])  # illustrative values
gap = discrimination_gap(oracle_risk, left, metric=lambda scores: roc_auc_score(labels, scores))
assert np.isclose(gap.gap, -1 / 9)
```

For differences across independent simulated datasets, use
[`compare_paired_replicates`](evaluation.md#paired-procedure-comparisons) instead.

`discrimination_gap` measures how much discrimination the fitted predictor loses
relative to the generating score on the evaluation sample. It reports the AUC of the
generating score minus the fitted AUC (`oracle_auc - fitted_auc`), using one supplied
metric and shared outcomes, censoring reference and evaluation sample. Negative
sample gaps are possible. `learnability_gap` is an alias. To bootstrap this gap,
return **fitted, then generating score** from the callback, so right-minus-left
equals the gap.
For a protective effect (`odds_ratio < 1` or `hazard_ratio < 1`) pass
`calibration["signal_coefficient"] * truth.signal`
as `oracle_risk`, so higher scores still mean greater event risk.
