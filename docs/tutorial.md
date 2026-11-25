# Benchmarking feature selection

Suppose a model predicts an outcome well and ranks its features by importance. You want to
know whether its top-ranked features generated the outcome, or are correlates: features
correlated with the generating ones, which predict almost as well when the correlation is
strong. On real data the generating features are usually unknown, so the question cannot be
answered.
PyPlasmode makes it answerable by keeping your measurements and simulating the outcome from
features you choose.

For your own study, replace the synthetic measurements below with a numeric matrix that has one
observation per row and one uniquely named feature per column. Install
[PyPlasmode](https://pypi.org/project/pyplasmode/), which brings NumPy and scikit-learn, and run
the blocks in order, because each uses names defined by the ones before it.

## Create correlated measurements

To show whether a model picks generating features or their correlates, the example matrix
contains two correlated pairs.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pyplasmode as ppm

rng = np.random.default_rng(17)
X = rng.normal(size=(4_000, 8))
X[:, 1] = 0.95 * X[:, 0] + np.sqrt(1 - 0.95**2) * X[:, 1]
X[:, 3] = 0.95 * X[:, 2] + np.sqrt(1 - 0.95**2) * X[:, 3]
names = ("A", "A_proxy", "B", "B_proxy", "C", "D", "E", "F")
population = ppm.Population(X, names)
```

The matrix starts as eight independent standard normal features. The two
assignments mix each of columns 1 and 3 with the column before it, in proportions that keep
unit variance and give a correlation of 0.95. A_proxy is now a close stand-in for A, B_proxy
for B, and C to F stay unrelated to everything. `Population` stores the matrix with its
column names. The seed 17, like every seed on this page, fixes the random draws so that you
see the numbers shown here.

A model may select A_proxy in place of A and lose little prediction. Whether that counts as finding
the signal depends on your question, so you declare which features belong together: each
`FeatureSet` names one group, and `FeatureSets` collects the groups. The signal below takes its
generating features from these groups, and recovery can then be scored by group as well as by
feature.

```python
groups = ppm.FeatureSets(
    (
        ppm.FeatureSet("A_group", ("A", "A_proxy")),
        ppm.FeatureSet("B_group", ("B", "B_proxy")),
    )
)
aligned = groups.align(population)
print(aligned.mapped, aligned.unmapped)
# ('A', 'A_proxy', 'B', 'B_proxy') ()
```

`groups.align(population)` matches the group members against the matrix columns. `mapped`
lists the members it found and `unmapped` the ones it did not. On your own data, groups often
come from an annotation file, and a name in `unmapped` is a member that matches no column, because
it is absent or spelt differently.

## Generate an outcome from known features

Now choose which features generate the outcome. Here they are A and B. Their correlates carry
information about the outcome only through their correlation with A and B, which is what makes
them a trap for a model's ranking.

```python
partition = ppm.partition_population(
    population, training_fraction=0.6, validation_fraction=0.2, seed=7
)
samples = ppm.generate_partitioned(
    partition,
    truth=ppm.CorrelatedTruth(group_names=("A_group", "B_group"), sentinels=("A", "B")),
    feature_sets=groups,
    outcome=ppm.BinaryOutcome(probability=0.15, odds_ratio=2.0),
    sample_sizes=ppm.PartitionSampleSizes(2_400, 800, 800),
    sampling_method="without_replacement",
    seed=42,
)
train, test = samples.training, samples.evaluation
print(train.truth.direct_features)
# ('A', 'B')
```

`partition_population` first splits the 4,000 rows into training, validation and test pools
of 60%, 20% and 20%. Splitting before anything is simulated keeps every observation used to fit
the model out of the rows used to judge it.

`generate_partitioned` then draws a sample from each pool and simulates an outcome for every row,
calibrating the three samples together. The sample sizes equal the pool sizes, and drawing without
replacement uses each row once. Drawing with replacement instead lets you simulate a study larger
than your data.

- `CorrelatedTruth` builds the signal from the groups. One member of each group generates the
  outcome; the API calls it the group's *sentinel*, here A and B. The *generating score* is
  the number computed for each row from the generating features: here the sum of standardised
  A and B, rescaled to unit standard deviation.
- `BinaryOutcome(probability=0.15, odds_ratio=2.0)` gives the outcome to 15% of rows on average
  across the three samples and doubles its odds for each standard deviation of the generating score.

`train.truth` records what a model should find. Its `direct_features` are the features a
selection should contain: A and B.

## Fit, tune and explain the model

PyPlasmode does not fit models. You fit one with your own tools and hand back its predictions
and its importance ranking. Here it is a cross-validated logistic regression.

```python
model = GridSearchCV(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000)),
    param_grid={"logisticregression__C": np.logspace(-4, 4, 5)},
    cv=5,
    scoring="roc_auc",
)
model.fit(train.X, train.outcome.values)
prediction = ppm.evaluate_prediction(
    test.outcome.values, model.predict_proba(test.X)[:, 1], metric="roc_auc"
)
scores = np.abs(model.best_estimator_[-1].coef_[0])
ranking = tuple(names[i] for i in np.argsort(-scores))
exact = ppm.evaluate_ranking(ranking, train.truth, depths=(2,)).at_depth[0]
group = ppm.evaluate_groups(ranking, train.truth, depths=(2,)).at_depth[0]
print(round(prediction.value, 3))
print(ranking)
print(exact.recall, exact.precision, group.group_recall)
# 0.686
# ('B', 'A', 'B_proxy', 'A_proxy', 'F', 'E', 'C', 'D')
# 1.0 1.0 1.0
```

1. `GridSearchCV` tries five regularisation strengths from 10⁻⁴ to 10⁴ and keeps the one with the
   best five-fold cross-validated AUC on the training sample. The scaler sits inside the pipeline so
   that it is [refitted within each
   fold](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage); fitted once on all
   training rows, it would carry information from each fold's held-out rows into the fit. The
   winning pipeline is then refitted on all training rows. Because cross-validation does the tuning,
   the validation sample is not needed.
2. `evaluate_prediction` scores the model's predicted probabilities on the test sample, which
   played no part in fitting or tuning.
3. The absolute fitted coefficients serve as importance scores. The inputs are standardised,
   so they compare features on one scale. `ranking` orders the features from most to least
   important. Any model's importance scores work the same way from here on, provided they are finite
   and not negative; take absolute values of signed scores.
4. `evaluate_ranking` and `evaluate_groups` score the top two features of the ranking against
   `train.truth`. `depths=(2,)` asks for the top two, and `at_depth[0]` picks that
   result.

The test-set AUC, the area under the ROC curve, is 0.686: the probability that the model gives
a randomly chosen row with the outcome a higher prediction than one without it, where 0.5 is
guessing. The model ranks B and A first. Recovery (`recall` in the API) is the fraction of the
generating features in the top two, here both. Precision is the fraction of the top two that
are generating features, also 1.0. Group recovery counts a group as found when any of its
members is selected.

The three measures separate when a model prefers a correlate. A top two of A and B_proxy would
have recovery and precision of 0.5, because only A generated the outcome, but group recovery
of 1.0, because B_proxy stands in for B's group.

## Compare with matched chance

A list of two features contains a generating feature some of the time by luck. *Recovery above
chance* is the recovery you observed minus the recovery a random list would get.

The answer depends on what "random" means. If one kind of feature is more likely to be a
generating feature, a list that favours that kind scores well without having identified
anything. A *stratum* is a label you give each feature to say which features the random list
may swap for which, so that the random list resembles the real one in kind. Here the labels
are `pair` for the four paired features and `singleton` for the other four.

```python
nomination = ppm.nomination_from_scores(names, scores, depth=2)
strata = ("pair", "pair", "pair", "pair", "singleton", "singleton", "singleton", "singleton")
matched = ppm.evaluate_matched_recovery(nomination, train.truth, estimand="exact", strata=strata)
print(matched.observed_recovery, matched.expected_recovery, matched.chance_adjusted_recovery)
# 1.0 0.5 0.5

unmatched = ppm.evaluate_matched_recovery(
    nomination, train.truth, estimand="exact", strata=("all",) * 8
)
print(unmatched.expected_recovery, unmatched.chance_adjusted_recovery)
# 0.25 0.75
```

`nomination_from_scores` turns the importance scores into the top two features (the API calls
a selection a *nomination*). `evaluate_matched_recovery` then:

1. counts the model's selections in each stratum: two in `pair`, none in `singleton`;
2. builds the random reference by keeping those counts and reshuffling which features within
   each stratum are selected;
3. computes the expected recovery of that reference exactly, without simulation;
4. returns the observed recovery, the expected recovery and their difference
   (`chance_adjusted_recovery`).

The model selected A and B, so its recovery is 1.0. What that is worth depends on the
reference:

| Random reference | Expected recovery | Recovery above chance |
| --- | --- | --- |
| One stratum: any two of the eight features | 0.25 | 0.75 |
| `pair` and `singleton`: any two of the four paired features | 0.50 | 0.50 |

Both selections were paired features, so the fair reference also picks two paired features,
and half of those are generating. The one-stratum 0.75 credits the model for looking among
correlated features at all. The matched 0.5 credits it only for telling A from A_proxy. Had
the model selected A and B_proxy, recovery would be 0.5 and recovery above chance zero.

## Matching strata on empirical measurements

Here the strata were known, because the pairs were built by hand. Real measurements carry no
such labels, so you need a rule that sorts features into strata. One rule is how strongly
each feature correlates with any other: features with close correlates are the ones a model
can swap for one another.

```python
correlations = np.abs(np.corrcoef(population.X, rowvar=False))
np.fill_diagonal(correlations, 0.0)
strongest = correlations.max(axis=0)
print(np.round(strongest, 2))
# [0.95 0.95 0.95 0.95 0.02 0.01 0.02 0.02]

labels = ("weak", "moderate", "strong")
empirical_strata = tuple(labels[i] for i in np.digitize(strongest, [0.3, 0.7]))
print(empirical_strata)
# ('strong', 'strong', 'strong', 'strong', 'weak', 'weak', 'weak', 'weak')

empirical = ppm.evaluate_matched_recovery(
    nomination, train.truth, estimand="exact", strata=empirical_strata
)
print(empirical.chance_adjusted_recovery, empirical.admissible_panel_count)
# 0.5 6
```

The first three lines compute the absolute correlation between every pair of features,
set each feature's correlation with itself to zero, and keep each feature's strongest
correlation with another. `np.digitize` then sorts features into three strata: below 0.3,
0.3 to 0.7, and 0.7 or above. The rule recovers the hand-made strata, so the result is the same.

Another rule is the size of the correlated group a feature belongs to. When a group counts as
recovered if any member is selected, a large group is easier to hit by luck, so features in
groups of similar size are compared with each other.

Three rules for choosing strata:

- **Use the measurements only.** Never use the outcomes, the signal or anything a fitted
  model produced, or the reference can be tuned, knowingly or not, to flatter a method.
- **Fix the rule before you look at results**, cut points included, for the same reason.
- **Keep strata coarse enough that chance can vary.** `admissible_panel_count` is the number
  of ways the random reference can rearrange the selection within its strata: 6 here, the
  ways to choose two of four paired features. If it is 1, the only possible random list is
  the observed one, and recovery above chance is zero whatever the model did.

`np.corrcoef` returns missing values for a column that is constant or contains missing
values. Drop constant columns first, and compute correlations on complete rows or with
pairwise deletion (for example `pandas.DataFrame.corr`).

## Change the signal structure

So far the outcome came from A and B alone, and only A and B counted as correct. An outcome
may instead depend on what a group of correlated features measures in common, and then a
model that selects A_proxy has found part of the signal. The signal structure decides both
which features generate the outcome and what a selection has to contain to recover it.

`materialize_truth` computes the generating score and the recovery targets from the
measurements without simulating an outcome, which is enough to see what each structure asks
of a selection. The example scores the model's top two, A and B, against two structures.

```python
sparse = ppm.materialize_truth(ppm.SparseTruth(features=("A", "B")), population, seed=3)
distributed = ppm.materialize_truth(
    ppm.CorrelatedTruth(group_names=("A_group", "B_group"), signal_form="group_means"),
    population,
    feature_sets=groups,
    seed=3,
)
print(sparse.direct_features, distributed.direct_features)
# ('A', 'B') ()
sparse_found = ppm.evaluate_ranking(ranking, sparse, depths=(2,)).at_depth[0]
groups_found = ppm.evaluate_groups(ranking, distributed, depths=(2,)).at_depth[0]
weight_found = ppm.evaluate_module(ranking, distributed, depths=(2,)).at_depth[0]
print(sparse_found.recall, groups_found.group_recall, weight_found.weighted_coverage)
# 1.0 1.0 0.5
```

`SparseTruth` generates the outcome from A and B directly, so A and B are the whole target
and their recovery is 1.0. `signal_form="group_means"` generates it from the mean of each
group's standardised features, so all four paired features contribute equally. No single
feature is the generating one, which leaves `direct_features` empty; `evaluate_ranking`
would raise an error. Recovery is measured in two other ways instead:

- `evaluate_groups` asks whether each group has a member selected. A and B find both groups.
- `evaluate_module` asks what share of the total generating weight the selection carries. A
  and B carry half, because A_proxy and B_proxy carry the other half.

The `seed` argument is required but matters only when the library picks features for you or a
custom signal draws random numbers.

To benchmark a method on either structure, pass the specification as `truth` to
`generate_partitioned` and fit again. Repeat the comparison with independent seeds, because one
simulated dataset gives one draw of each result, and report prediction, precision and recovery
across the repeats. Include a null signal (`NullTruth`), because a method that selects the same
features when nothing generates the outcome is selecting by something other than the signal, such as
structure in the measurements or a fixed preference of its own. For continuous responses, counts,
ordered categories and time-to-event data, see [Outcomes](outcomes.md).

## Choose discrimination and compare refits

The comparison so far used one effect size, one fit and one dataset, so it cannot show whether
the result holds beyond them. Each of these extends it:

- **Set difficulty as an AUC.** `BinaryOutcome(probability=0.15, target_auc=0.7)` solves for
  the odds ratio that gives the generating score an expected AUC of 0.7, so you choose how
  predictable the outcome is in the same unit you report for the model.
  [AUC calibration](calibration.md#what-is-solved)
- **Check that a selection is stable.** Refits on resampled training data can rank different
  features first. [`selection_stability`](metrics.md#selection-stability) measures how much
  their selections agree, and [`consensus_ranking`](metrics.md#consensus-and-rank-biased-overlap)
  combines their importance scores into one ranking.
- **Compare two fitted models on the same test rows.**
  [`paired_bootstrap_difference`](metrics.md#paired-bootstrap-and-discrimination-gap) resamples
  the shared rows for both models together, so the interval for their difference reflects that
  both were scored on the same observations.
- **Compare two procedures in general.** Run both on each independently simulated dataset and
  pass the results to [`compare_paired_replicates`](evaluation.md#paired-procedure-comparisons),
  which pairs them by dataset before taking differences.
