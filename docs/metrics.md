# Metrics

`pyplasmode.metrics` measures agreement between repeated feature selections, AUC (area under the
ROC curve) under censoring and differences between two methods on the same observations. Its
functions take arrays and selections without fitting models. Align features and observations
before using an array-based metric.

## Selection stability

`selection_stability` measures how consistently refits select the same features. A
`value` of 1 means identical selections with no split slots, 0 means chance-level
agreement, and a negative value means less agreement than chance.

Use [`nomination_from_scores`](evaluation.md#rankings-and-scores) to make one selection
per refit. It splits a slot among features tied at the cutoff and excludes zero
importance. Supply at least two refits that score the same two or more features.

```python
import pyplasmode as ppm
from pyplasmode.metrics import selection_stability, consensus_ranking, rank_biased_overlap

features = ("A", "B", "C", "D")
scores = [[4, 3, 0, 0], [4, 0, 0, 0], [4, 0, 3, 0]]
nominations = [ppm.nomination_from_scores(features, row, depth=2) for row in scores]
stability = selection_stability(nominations)
print(round(stability.value, 3))  # 0.314
print(stability.frequencies)  # (1.0, 0.3333333333333333, 0.3333333333333333, 0.0)
print(stability.membership_budgets)  # (2.0, 1.0, 2.0)
```

A is selected every time, B and C once each, and D never. Five slots across
12 refit–feature combinations give `q = 5/12`.

For each feature, let `Q` be its selection probabilities across refits. Compute
`var(Q, ddof=1) + mean(Q*(1-Q))`, then average over features. Stability is one
minus this average divided by the chance variance `q*(1-q)`. The second term
accounts for ties being broken independently in each refit. When every refit ties all features for
the same number of slots, that term alone equals the chance variance and
stability is zero whenever only some slots are filled. The measure extends Definition 4 of
[Nogueira et al.](https://jmlr.org/papers/v18/17-514.html).

If every selection is empty or every selection is full, `value` is `None` because
the chance variance is zero. When refits fill different numbers of slots, stability
is below one even if every smaller selection is contained in every larger one.

## Consensus and rank-biased overlap

Use `consensus_ranking(feature_ids, scores, depth=...)` to order features across
refits. It sorts by average selection probability at that depth, highest first.
It breaks ties using the median rank among refits that gave the feature any
membership. Within each refit, ranks start at 1 and ties share the average of
their positions across all features. A feature never selected has median rank
`float("inf")`.

`consensus.ranking` lists feature names in sorted order, alphabetically when the
evidence ties. `consensus.ranks`, `inclusion_probabilities` and
`median_selected_ranks` follow the order of the input `feature_ids`. Features with
equal evidence have equal numerical ranks.

`rank_biased_overlap` (RBO) measures how much two rankings agree near the top.
Pass complete rank vectors for the same features in the same order, such as
the `ranks` from two consensus results built with identical `feature_ids`. Lower
ranks are better. Convert importance scores with
`scipy.stats.rankdata(-scores, method="average")` for a NumPy array of scores.

- Use `tie_method="webber"` (default) to count a tie as a group. Include the whole
  group if it starts at or above the cutoff. Agreement is twice the number of
  shared features divided by the combined size of the two lists.
- Use `tie_method="independent"` to compare the lists you would get by breaking
  ties independently and uniformly at random. Agreement is the expected number
  of shared features divided by the number of slots.

```python
consensus = consensus_ranking(("C", "B", "A"), [[0, 4, 4], [0, 3, 3]], depth=1)
print(consensus.ranking)  # ('A', 'B', 'C')
print(consensus.ranks)  # (3.0, 1.5, 1.5)
for tie_rule in ("webber", "independent"):
    print(
        rank_biased_overlap(consensus.ranks, consensus.ranks, persistence=0.5, tie_method=tie_rule)
    )
# 1.0
# 0.75
```

Both rules give full agreement from depth two. At depth one, the whole-tie rule
keeps both A and B, giving full agreement. With independent tie breaks, the
first features match half the time. Depth one has weight 0.5, so RBO is
`0.5 * 0.5 + 0.5 * 1 = 0.75`.

For `n` features and `p = persistence` (default 0.9), depth `d` gets weight
`(1-p)*p**(d-1)` for depths 1 to `n-1`. The remaining weight, `p**(n-1)`,
goes to the full list, whose agreement is always one. See Equation 28 of
[Webber et al.](https://www.codalism.com/research/papers/wmz10_tois.pdf) for the
whole-tie convention and [`rank_stability`](evaluation.md#ranking-stability)
for Jaccard, Spearman and RBO on untied rankings.

## Censored discrimination

`cumulative_dynamic_auc` measures how well risk scores separate people who had
an event by `horizon` (cases) from people observed beyond it (controls).
People censored at or before the horizon are neither. Supply risk scores that
increase with event risk.

AUC is the weighted fraction of case–control pairs in which the case ranks
higher, with half credit for ties. Each case gets weight `1/G(t)` at its event
time `t`. Here `G(t)` is the probability of remaining uncensored past `t`,
estimated by reverse Kaplan–Meier on `training_time` and `training_event`.
When an event and censoring occur at the same training time, events leave the
risk set first. Controls all have the same weight, which cancels in the fraction.

```python
from pyplasmode.metrics import cumulative_dynamic_auc

auc = cumulative_dynamic_auc(
    training_time=[1, 2, 4, 5],
    training_event=[1, 0, 1, 0],
    time=[1, 2, 4, 5],
    event=[1, 0, 1, 0],
    risk=[0.9, 0.2, 0.1, 0.0],
    horizon=3,
)
print(auc)  # 1.0
```

The first row is the only case and outranks both controls, the third and fourth
rows. The second row is excluded because follow-up ended at time 2 without an
observed event.

By default, `tied_tolerance=0.0` gives half credit only for exact ties. With a
positive tolerance, consecutive scores within that distance form one tie block,
even when the endpoints are farther apart. Set `tied_tolerance=1e-8` to match the
default tie rule of [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.cumulative_dynamic_auc.html).

Use the same time unit throughout. For ten years in days, use
`horizon=ppm.years(10).days`. The horizon must be finite and positive, below
`max(training_time)`, and below at least one evaluation time. See
[calibration](calibration.md) to generate longer follow-up. For landmarking,
keep only rows observed past the landmark in both samples, then subtract the
landmark from their times and the absolute horizon.

Invalid inputs, missing cases or controls, or zero censoring survival at a case
time or at the horizon raise `ValueError`. The estimator assumes independent
censoring and does not handle competing risks or delayed entry.

## Paired bootstrap and discrimination gap

Use `paired_bootstrap_difference(participants, statistic)` to get a percentile
interval for the difference between two fixed models evaluated on the same rows.
Pass a row count or unique string IDs in row order. The callback `statistic`
receives zero-based integer row indices either way. Each draw samples the original
number of rows with replacement. Use the same indices for both statistics
and return `(left, right)`. The callback also determines whether to re-estimate
quantities used to score the models, such as censoring weights.

`left` and `right` give the original estimates, and `difference` is right minus
left on those rows. `confidence_interval_low` and `confidence_interval_high` give
the percentile limits. Defaults are 1,000 attempts, 95% confidence and seed 0.

Return `None` if a draw makes either statistic undefined. Those draws appear as
`None` in `differences` and are counted in `failed_resamples`. The original
statistics must be defined and at least two draws must succeed. The interval is
conditional on success, so report exclusions such as zero-event draws.

`discrimination_gap` applies one AUC metric to the generating and fitted
scores on the same evaluation data. It returns `oracle_auc`, `fitted_auc` and
their difference, `gap = oracle_auc - fitted_auc`. The gap is negative when
the fitted AUC is higher.

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


comparison = paired_bootstrap_difference(len(labels), statistic, resamples=100, seed=7)
print(comparison.difference, comparison.failed_resamples)  # 0.0 5

oracle_risk = np.array([0.1, 0.2, 0.4, 0.8, 0.5, 0.9])  # illustrative values
gap = discrimination_gap(oracle_risk, left, metric=lambda scores: roc_auc_score(labels, scores))
print(round(gap.gap, 3))  # -0.111
```

The generating score ranks eight of nine case–control pairs correctly: the case
at 0.4 ranks below the control at 0.5. The fitted score ranks all nine correctly,
so the gap is `8/9 - 1 = -1/9`.

For a protective effect, such as an odds or hazard ratio below one, pass
`outcome.calibration["signal_coefficient"] * truth.signal` as `oracle_risk` so that
higher scores mean greater event risk. `learnability_gap` is an alias. To
bootstrap the gap, return the fitted AUC first and the generating-score AUC
second. To compare methods across independent simulated datasets, use
[`compare_paired_replicates`](evaluation.md#paired-procedure-comparisons).
