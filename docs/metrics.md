# Metrics

The functions in `pyplasmode.metrics` answer three questions that come up once you have fitted
models: whether repeated fits select the same features, how well risk scores separate people
with and without an event when follow-up is censored, and whether one fitted model beats another
on the same people. They fit nothing themselves. Array arguments are matched by position, so
give features and rows in the same order in every argument; a misaligned array gives a wrong
number, not an error.

## Selection stability

Selection stability answers one question: if you refit a selection method on resampled data,
does it select the same features each time, more than chance would?

**How it is used.** Make one selection per refit with
[`nomination_from_scores`](evaluation.md#rankings-and-scores), which shares a slot among tied
features and never selects a zero score, and pass them to `selection_stability`. It then:

1. measures how much each feature's selection varies across refits, including the variation a
   tie adds once broken;
2. compares that with the variation if the same number of slots were filled at random,
   extending Definition 4 of [Nogueira et al.](https://jmlr.org/papers/v18/17-514.html);
3. returns `value`: 1 when every refit selected the same features with no tie at the cutoff, 0
   when they agree no more than random selections, negative when less.

Here three refits score four features and each asks for two:

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

for row in ([4, 3, 0, 0], [1, 1, 1, 1]):
    repeated = [ppm.nomination_from_scores(features, row, depth=2)] * 3
    print(row, selection_stability(repeated).value)
# [4, 3, 0, 0] 1.0
# [1, 1, 1, 1] 0.0
```

| Three refits | Selections | `value` |
| :-- | :-- | --: |
| Same scores each time | A and B every time | 1.0 |
| The example | A and B; A alone; A and C | 0.314 |
| Every feature tied each time | two of four at random | 0.0 |

`frequencies` gives each feature's selection rate: A every time, while the second slot goes to
B, to nobody (`membership_budgets` shows one slot filled) or to C. The tied method's refits are
identical, yet it scores zero, as random selection should, because each tie is broken afresh.

- **At least two refits over the same two or more features are needed**, or the call raises
  `ValueError`.
- **`value` is `None` when every refit selects nothing or every feature**, because random
  selection then cannot vary either.
- **Refits that fill different numbers of slots score below 1**, even when the selections are
  nested, so a method that sometimes gives fewer than `depth` features a positive score is
  penalised.

## Consensus and rank-biased overlap

`consensus_ranking` answers one question: after several refits, which single ranking do you
report? `rank_biased_overlap` answers the next: how much do two rankings agree?

**How the consensus is built.** `consensus_ranking(feature_ids, scores, depth=...)` takes one row
of importance scores per refit. It then:

1. fills up to `depth` slots in each refit, as `nomination_from_scores` would;
2. puts features selected more often across refits first;
3. among features selected equally often, puts first the one with the better median rank in the
   refits that selected it (rank 1 is the highest score; tied scores share the average of their
   positions);
4. puts features never selected last.

It returns `ranking`, the names in order (alphabetical where evidence ties), and `ranks`, in
input order with equal ranks for equal evidence, so two results built from the same
`feature_ids` compare position by position.

**How the comparison is used.** Like [`rank_stability`](evaluation.md#ranking-stability), RBO
counts agreement near the top more heavily, but it accepts ties. Pass one rank per feature,
lower is better, in one feature order for both, such as two consensus results' `ranks`; for
scores, `scipy.stats.rankdata(-scores, method="average")` gives the highest rank 1.
`tie_method` sets how a tie at a cutoff counts:

- `"webber"` (default) includes the whole tie whenever it starts at or above the cutoff, and
  scores twice the shared features over the two lists' combined size. Use it when a tie means
  the method rates the features as equal, so two lists with the same tie agree fully.
- `"independent"` breaks every tie uniformly at random and scores the expected shared features
  over the number of slots. Use it when ties will be broken at random before the ranking is
  used, so agreement reflects the lists actually used.

Here two refits score A and B equally, above C, and comparing the consensus with itself
isolates the tie:

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

A and B share the single slot in both refits, so they share ranks 1 and 2 (1.5 each); C, never
selected, has rank 3. With `persistence=0.5`, depth one carries weight 0.5 and the rest, 0.5,
agrees fully:

| `tie_method` | Agreement at depth one | RBO |
| :-- | --: | --: |
| `"webber"` | 1: both lists hold A and B | 1.0 |
| `"independent"` | 0.5: first features match half the time | 0.5 × 0.5 + 0.5 × 1 = 0.75 |

A `persistence` below its default of 0.9 moves weight to the first few depths. Without ties RBO
equals the score from `rank_stability`. The whole-tie rule follows
Equation 28 of [Webber et al.](https://www.codalism.com/research/papers/wmz10_tois.pdf).

## Censored discrimination

Censored discrimination answers one question: for a time-to-event outcome, do the risk scores
separate people who have an event by a chosen time, the `horizon`, from people who do not?

**Why it is needed.** People who leave follow-up before the horizon without an event are
*censored*: whether they would have had one is unknown. Leaving them out biases the result,
because they include people who would have had a late event, so cases with late event times are
under-represented.

**How it is used.** You pass the training follow-up (`training_time`, `training_event`), the
evaluation rows' `time`, `event` and `risk` (increasing with event risk, or the AUC is
reversed), and the `horizon`. `cumulative_dynamic_auc` then:

1. takes as *cases* people with an event at or before the horizon, and as *controls* people
   with none by the horizon and followed after it; people censored by the horizon are left out;
2. estimates `G(t)`, the probability of remaining uncensored past `t`, by reverse Kaplan-Meier
   from the training rows;
3. weights each case by `1/G(t)` at its event time `t`; controls share one weight, which cancels;
4. returns AUC (the area under the ROC curve): the weighted share of case-control pairs in which
   the case has the higher risk, with half credit for a tie.

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

The example reuses the training rows as evaluation rows to stay short; in a benchmark, pass the
held-out rows. At horizon 3:

| Row | Time | Event | Risk | Role |
| --: | --: | --: | --: | :-- |
| 1 | 1 | yes | 0.9 | case |
| 2 | 2 | no | 0.2 | left out: censored at 2 |
| 3 | 4 | yes | 0.1 | control: event after 3 |
| 4 | 5 | no | 0.0 | control |

The case's 0.9 is above both controls' risks, so AUC is 1.

- **Choose a horizon the AUC can be estimated at**, or the call raises `ValueError`: below the
  longest training follow-up, because `G` cannot be estimated beyond it; before the last
  evaluation time, so there are controls; with `G` above zero at every case's event time and at
  the horizon, because a case would otherwise get infinite weight.
- **Give `time`, `training_time` and `horizon` in one unit**: for ten years in days,
  `horizon=ppm.years(10).days`. If simulated follow-up is too short, see
  [calibration](calibration.md).
- **Ties:** by default (`tied_tolerance=0.0`) only exactly equal scores get half credit. With a
  positive tolerance, consecutive sorted scores within it form one tie, so a chain of close
  scores can tie endpoints farther apart. `tied_tolerance=1e-8` treats ties as
  [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.cumulative_dynamic_auc.html)
  does.
- **For a later starting time (a landmark)**, keep rows followed beyond it in both samples and
  subtract it from their times and the horizon, since the function measures time from zero.
- **The weighting assumes censoring unrelated to event risk.** Competing risks and late entry
  to follow-up (delayed entry) are not handled.

## Paired bootstrap and discrimination gap

`paired_bootstrap_difference` answers one question: would the difference between two fitted
models' scores on the same people hold in another sample of people? `discrimination_gap`
answers another: how much discrimination does a fitted model leave unrealised?

**How the bootstrap is used.** `participants` is the number of rows or their unique string IDs
in row order. You write `statistic`: it receives an array of 0-based row indices and returns
`(left, right)`, the two models' scores on those rows. `paired_bootstrap_difference` then:

1. draws the original number of rows with replacement (1,000 times by default, seed 0);
2. scores both models on each draw, so the variation from which people were drawn moves both
   scores together and, as far as it does, cancels in the difference;
3. records the difference, right minus left; a draw where `statistic` returns `None` (for
   example one with no events, for AUC) is recorded as `None` in `differences`, counted in
   `failed_resamples` and left out of the interval;
4. returns the original `left`, `right` and `difference`, and a percentile interval (95% by
   default).

**How the gap is used.** The generating score is the risk score the simulation used, so,
oriented so that higher means greater risk, its AUC is the most a fitted model can expect on
average. `discrimination_gap` applies one AUC function to both on the same evaluation data and
returns `oracle_auc`, `fitted_auc` and the gap, `oracle_auc - fitted_auc`, negative when the
fitted model does better by chance in a finite sample.

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

Both models rank every case above every control, so their difference is 0. Of the 100 draws
(few, to run quickly), five held one class only and returned `None`. `oracle_risk` stands in for
a generating score: it ranks eight of nine case-control pairs correctly (the case at 0.4 falls
below the control at 0.5) and `left` all nine, so the gap is `8/9 - 1 = -1/9`.

- **Use the same indices for both scores**, so the comparison stays paired.
- **Recompute inside `statistic` anything estimated from the evaluation rows**, so the interval
  includes its uncertainty. `cumulative_dynamic_auc` takes its censoring weights from the
  training rows, which stay fixed.
- **Report how many draws failed**, since the interval describes only the successes. The
  original scores must be defined, and at least two draws must succeed.
- **For a protective signal**, such as an odds or hazard ratio below one, the generating score is
  higher for people at lower risk, and its AUC would be expected to fall below 0.5. Pass
  `outcome.calibration["signal_coefficient"] * truth.signal` as `oracle_risk`, so that higher
  scores mean greater event risk.
- **To bootstrap the gap**, return the fitted AUC first and the generating score's AUC second;
  the difference, right minus left, is then the gap.
- **To compare methods across independent simulated datasets**, use
  [`compare_paired_replicates`](evaluation.md#paired-procedure-comparisons). The bootstrap holds
  the fitted models fixed, so it omits the variation from refitting on new data.
