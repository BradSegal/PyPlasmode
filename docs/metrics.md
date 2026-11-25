# Metrics

The functions in `pyplasmode.metrics` answer three questions that come up once you have fitted
models: whether repeated fits select the same features, how well risk scores separate people
with and without an event when follow-up is censored, and whether one fitted model beats another
on the same people. They take scores, selections and outcomes you have already computed and fit
nothing themselves. Array arguments are matched by position, so give features and rows in the
same order in every argument; a misaligned array gives a wrong number, not an error.

## Selection stability

If you refit a selection method on resampled data, a reliable method selects mostly the same
features each time. `selection_stability` turns that into one number corrected for chance. A
`value` of 1 means every refit selected the same features, 0 means the refits agree no more
than random selections of the same size would, and a negative value means they agree less.

Make one selection per refit with
[`nomination_from_scores`](evaluation.md#rankings-and-scores), which shares a slot among
features tied at the cutoff and never selects a feature with zero importance. Here three refits
score four features and each selects two:

```python
import pyplasmode as ppm
from pyplasmode.metrics import selection_stability, consensus_ranking, rank_biased_overlap

features = ("A", "B", "C", "D")
scores = [[4, 3, 0, 0], [4, 0, 0, 0], [4, 0, 3, 0]]
nominations = [ppm.nomination_from_scores(features, row, depth=2) for row in scores]
stability = selection_stability(nominations)
print(round(stability.value, 3))
# 0.314
print(stability.frequencies)
# (1.0, 0.3333333333333333, 0.3333333333333333, 0.0)
print(stability.membership_budgets)
# (2.0, 1.0, 2.0)
```

The first refit selects A and B and the third A and C. The second gives only A a positive score,
so it selects A alone. `frequencies` is each feature's selection rate across refits: A every
time, B and C once each, D never. `membership_budgets` is the number of slots each refit
filled. The value of 0.314 is well below 1 because the second slot goes to a different feature,
or to none, in each refit.

The measure compares how much each feature's selection varies across refits with how much it
would vary if the same number of slots were filled at random. Let `Q` be one feature's
membership in each refit, and `q` the mean membership over all refits and features: here 5
filled slots over 12 refit-feature combinations, `q = 5/12`. For each feature compute
`var(Q, ddof=1) + mean(Q * (1 - Q))`, then average over features. Stability is one minus this
average divided by `q * (1 - q)`, the variation of random selection. The second term counts
the variation that a tie adds once it is broken: a feature with membership 0.5 in every refit is
selected in half of them, which is instability. So a method that ties every feature in every
refit and fills its slots at random scores zero, as random selection should. The measure
extends Definition 4 of [Nogueira et al.](https://jmlr.org/papers/v18/17-514.html).

- **At least two refits over the same two or more features are needed**, since the measure
  compares refits and features with one another; otherwise the call raises `ValueError`.
- **`value` is `None` when every refit selects nothing or every refit selects every feature**,
  because random selection then cannot vary either, and there is nothing to compare with.
- **Refits that fill different numbers of slots score below 1** even when every smaller
  selection is contained in every larger one, as the second refit here shows. A method that
  sometimes gives fewer than `depth` features a positive score is penalised for it.

## Consensus and rank-biased overlap

After refitting several times you usually want one ranking to report or to carry forward.
`consensus_ranking(feature_ids, scores, depth=...)` combines the refits. Each row of `scores`
holds one refit's importance scores, and each refit selects `depth` features as
`nomination_from_scores` would. Features selected more often across refits come first. Among
features selected equally often, the one ranked higher in the refits that selected it comes
first, measured by its median rank in those refits. A feature that is never selected has a
median rank of `float("inf")` and comes last. Within a refit, rank 1 is the highest score, and
tied scores share the average of their positions.

`consensus.ranking` lists the names in consensus order, alphabetically where the evidence
ties. `consensus.ranks`,
`inclusion_probabilities` and `median_selected_ranks` follow the order of the input
`feature_ids`, and features with equal evidence get equal ranks, so that two consensus results
built from the same `feature_ids` can be compared position by position.

To compare two such rankings, use `rank_biased_overlap` (RBO). Like `rank_stability` on the
[Evaluation](evaluation.md#ranking-stability) page, it measures how much two rankings agree,
counting agreement near the top more heavily, but it accepts ties. It takes rank vectors, not
names: one rank per feature, lower is better, the same features in the same order in both. The
`ranks` of two consensus results fit directly. To convert a NumPy array of importance scores,
use `scipy.stats.rankdata(-scores, method="average")`, which gives the highest score rank 1.

A tie at a cutoff can be counted in two ways, chosen with `tie_method`:

- `"webber"` (default) includes the whole tie whenever it starts at or above the cutoff, and
  scores agreement as twice the number of shared features divided by the combined size of the
  two lists. Use it when a tie means the method rates the features as equal, so that two lists
  with the same tie agree fully.
- `"independent"` scores the lists you would get by breaking every tie uniformly at random, and
  takes the expected number of shared features divided by the number of slots. Use it when ties
  will be broken at random before the ranking is used, so that agreement reflects the lists that
  will actually be used.

Here two refits score A and B equally, both above C. Comparing the consensus with itself
isolates the effect of the tie:

```python
consensus = consensus_ranking(("C", "B", "A"), [[0, 4, 4], [0, 3, 3]], depth=1)
print(consensus.ranking)
# ('A', 'B', 'C')
print(consensus.ranks)
# (3.0, 1.5, 1.5)
for tie_rule in ("webber", "independent"):
    print(
        rank_biased_overlap(consensus.ranks, consensus.ranks, persistence=0.5, tie_method=tie_rule)
    )
# 1.0
# 0.75
```

A and B share the single slot in both refits, so they have equal evidence: they share ranks 1
and 2, giving each 1.5, and `ranking` lists them alphabetically. C, in first position of the
input, is never selected and has rank 3. With the whole-tie rule both lists include A and B at
depth one, so agreement is full at every depth and RBO is 1. With independent tie breaks, the
first features match only half the time. With `persistence=0.5` depth one has weight 0.5 and
the remaining weight, 0.5, has full agreement, so RBO is `0.5 * 0.5 + 0.5 * 1 = 0.75`.

In general, for `n` features and persistence `p` (`persistence`, default 0.9), the agreement at
depth `d` gets weight `(1-p) * p**(d-1)` for depths 1 to `n-1`. The remaining weight, `p**(n-1)`,
goes to the full list, whose agreement is always one. A smaller `p` concentrates the weight on
the first few depths. For rankings without ties this equals the score from `rank_stability`.
The whole-tie rule follows Equation 28 of
[Webber et al.](https://www.codalism.com/research/papers/wmz10_tois.pdf).

## Censored discrimination

For a time-to-event outcome, you want to know whether the risk scores separate people who have
an event by a chosen time, the `horizon`, from people who do not. Some people leave follow-up
before the horizon without an event (they are *censored*), and whether they would have had one
is unknown. `cumulative_dynamic_auc` computes AUC (the area under the ROC curve) for this
setting:

- *cases* had an event at or before the horizon;
- *controls* had no event by the horizon and were still followed after it;
- people censored at or before the horizon are neither, and are left out.

AUC is the share of case-control pairs in which the case has the higher risk score, with half
credit for a tie. Supply risk scores that increase with event risk, otherwise the AUC is
reversed. Leaving out censored people would bias the result: they include people who would have
had a late event, so cases with late event times would be under-represented. To correct for
this, each case is weighted by `1/G(t)`, where `t` is its event time and `G(t)` is the
probability of remaining uncensored past `t`. `G` is estimated by reverse Kaplan-Meier from
`training_time` and `training_event`. Controls all get the same weight, which cancels.

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
print(auc)
# 1.0
```

With a horizon of 3, the first row is the only case: it has an event at time 1. The third and
fourth rows are controls, followed past time 3. The second row is left out, because its
follow-up ended at time 2 without an event. The case's risk of 0.9 is above both controls' risks,
so AUC is 1.

By default, `tied_tolerance=0.0` gives half credit only to exactly equal scores. With a positive
tolerance, consecutive sorted scores within that distance of each other form one tie, so a chain
of close scores can tie endpoints that are farther apart. Set `tied_tolerance=1e-8` to reproduce
the results of [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.cumulative_dynamic_auc.html),
which uses that tolerance.

The call raises `ValueError` when the AUC cannot be estimated:

- the horizon must be finite, positive and below the longest training follow-up, because `G`
  cannot be estimated beyond it;
- the evaluation data must contain at least one case and one control, since AUC compares them.
  A horizon at or beyond every evaluation time leaves no controls;
- `G` must be above zero at every case's event time and at the horizon, because a case would
  otherwise get infinite weight.

Give `time`, `training_time` and `horizon` in the same unit. For ten years measured in days,
use `horizon=ppm.years(10).days`. If simulated follow-up is too short for your horizon, see
[calibration](calibration.md) to generate longer follow-up. To measure discrimination from a
later starting time (a landmark), keep only rows still followed at the landmark in both samples,
then subtract the landmark from their times and from the horizon, since the function measures
time from zero.

The weighting assumes that censoring is unrelated to event risk. The function does not handle
competing risks or people who enter follow-up late (delayed entry).

## Paired bootstrap and discrimination gap

Two fitted models evaluated on the same people will have different AUCs, and you want to know
whether the difference would hold in another sample of people. `paired_bootstrap_difference`
answers this by redrawing the evaluation rows with replacement many times and recomputing both
models' scores on each draw. The draw is *paired*: both models are scored on the same rows, so
the variation that comes from which people were drawn affects both scores alike and largely
cancels in the difference.

In `paired_bootstrap_difference(participants, statistic)`, `participants` is the number of
rows or their unique string IDs in row order. You write `statistic`: it receives an array of
0-based row indices, one draw of the original number of rows, and returns `(left, right)`, the
two models' scores on those rows. Use the same indices for both scores, so the comparison stays
paired. Recompute inside the callback anything that depends on which rows are drawn, such as
censoring weights, so that the interval includes its uncertainty.

The result gives the original scores, `left` and `right`, and their `difference`, right minus
left. `confidence_interval_low` and `confidence_interval_high` are percentile limits of the
differences across draws. By default the function makes 1,000 draws, gives a 95% interval and
uses seed 0.

If a draw makes either score undefined, for example a draw with no events for AUC, return
`None`. That draw is recorded as `None` in `differences`, counted in `failed_resamples` and left
out of the interval. The interval then describes only the draws that succeeded, so report how
many failed. The scores on the original rows must be defined, and at least two draws must
succeed.

`discrimination_gap` answers a different question: how much discrimination a fitted model leaves
unrealised. The generating score is the risk score the simulation used, so its AUC is the most a
fitted model can expect on average. `discrimination_gap` applies one AUC function to the
generating score and to the fitted scores on the same evaluation data. It returns `oracle_auc`
(the generating score's AUC), `fitted_auc` and the gap, `oracle_auc - fitted_auc`. In a finite
sample the fitted model can do better by chance, and the gap is then negative.

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
print(comparison.difference, comparison.failed_resamples)
# 0.0 5

oracle_risk = np.array([0.1, 0.2, 0.4, 0.8, 0.5, 0.9])
gap = discrimination_gap(oracle_risk, left, metric=lambda scores: roc_auc_score(labels, scores))
print(round(gap.gap, 3))
# -0.111
```

Both models rank every case above every control, so their difference on the original rows is 0.
Five of the 100 draws contained only one class, for which AUC is undefined, so `statistic`
returned `None`. `oracle_risk` stands in for a generating score. It ranks eight of the nine
case-control pairs correctly, because the case at 0.4 falls below the control at 0.5, while the
fitted score `left` ranks all nine correctly. The gap is `8/9 - 1 = -1/9`.

- **For a protective signal**, such as an odds or hazard ratio below one, the generating score is
  higher for people at lower risk, and its AUC would fall below 0.5. Pass
  `outcome.calibration["signal_coefficient"] * truth.signal` as `oracle_risk`, so that higher
  scores mean greater event risk.
- **To bootstrap the gap**, return the fitted AUC first and the generating score's AUC second.
  The bootstrap difference, right minus left, is then the gap.
- **To compare methods across independent simulated datasets** rather than across the people in
  one, use [`compare_paired_replicates`](evaluation.md#paired-procedure-comparisons). The
  bootstrap holds the fitted models fixed, so it does not include the variation from refitting
  on new data.

`learnability_gap` is the same function under another name.
