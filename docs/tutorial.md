# Benchmarking feature selection

Suppose a model predicts well and ranks features by importance. Did the highest-ranked features
generate the outcome, or are they correlates, measured features strongly correlated with them?

The measurements are synthetic. For your own study, supply a numeric matrix with one observation
per row and one feature per column, with unique column names. Install
[PyPlasmode](https://pypi.org/project/pyplasmode/) (NumPy and scikit-learn are included). Run
the blocks in order.

## Create correlated measurements

A and B each have a correlated partner. C to F vary independently. `Population` holds the
matrix and column names. Make a named `FeatureSet` for each pair and collect them in
`FeatureSets` to define the signal and measure group recovery.

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
groups = ppm.FeatureSets(
    (
        ppm.FeatureSet("A_group", ("A", "A_proxy")),
        ppm.FeatureSet("B_group", ("B", "B_proxy")),
    )
)
print(groups.align(population).mapped)
# ('A', 'A_proxy', 'B', 'B_proxy')
```

`groups.align(population)` checks the group members against the matrix columns. `mapped` lists
the four grouped columns. `unmapped` flags group members missing from the matrix, and
`excluded` lists columns outside any group (C to F here).

## Generate an outcome from known features

Only A and B enter the generating score, the number computed for each row from the generating
features. A_proxy and B_proxy are their correlates and carry information about the outcome
without entering the score. Each group's generating member is its *sentinel*. Both appear in
`direct_features`. Here, the outcome has a 15% marginal probability and
its odds double for each standard-deviation increase in the generating score.

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

## Fit, tune and explain the model

Five-fold cross-validation selects the regularisation strength using the training sample.
The [scaler is fitted within each fold](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage),
then the chosen pipeline is refitted on all training
rows. The test sample is used only for evaluation, and the validation sample is unused
because tuning uses cross-validation.

Absolute coefficients rank features by their fitted effect per training standard deviation.
For another model, use its own importance scores with the same recovery functions.

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
print(prediction)
print(ranking[:2], exact, group)
```

The prediction result reports test-set AUC, the area under the ROC curve: how well the
model ranks positive outcomes above negative ones. Recovery (`recall` in the API) is the
fraction of A and B found. Precision is the fraction of selected features that are generating
features. Selecting A and B gives both scores 1.0. Selecting A and B_proxy gives both scores
0.5 and recovers both groups.

## Compare with matched chance

A list of two features contains a generating feature some of the time by luck. Recovery above
chance removes that luck: it is the recovery you observed minus the recovery expected from
choosing features at random.

The random choice has to be a fair one. Picking any two of the eight features finds a quarter
of A and B on average. But A and B are both among the four paired features, and a model's top
choices tend to fall there too, on a generating feature or on its correlate. Picking two of
the four paired features finds half of A and B on average, so a model can score well above
the all-features figure without telling A from A_proxy.

To make the comparison fair, give each feature a label that says what kind it is, called its
*stratum*: here `pair` for the four paired features and `singleton` for the other four. The
random reference keeps the number of selections in each stratum and varies only which features
within it are chosen.

```python
nomination = ppm.nomination_from_scores(names, scores, depth=2)
strata = ("pair", "pair", "pair", "pair", "singleton", "singleton", "singleton", "singleton")
matched = ppm.evaluate_matched_recovery(nomination, train.truth, estimand="exact", strata=strata)
print(matched.observed_recovery, matched.expected_recovery, matched.chance_adjusted_recovery)
# 1.0 0.5 0.5
```

`nomination_from_scores` turns the importance scores into the top two features (the API calls
a selection a *nomination*). The model selected A and B, so recovery is 1.0. Two paired
features chosen at random recover 0.5, which leaves a recovery above chance of 0.5
(`chance_adjusted_recovery`). Had the model selected A and B_proxy, recovery would be 0.5 and
recovery above chance zero: no better than choosing paired features at random.

## Matching strata on empirical measurements

Here the strata were known, because the pairs were built by hand. Real measurements carry no
such labels, so you need a rule that sorts features into strata. A workable rule is how
strongly each feature correlates with any other: features with close correlates are the
ones a model can swap for one another, and they should be compared with each other.

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
0.3 to 0.7, and above 0.7. The rule recovers the hand-made strata, so the result is the same.

Three things keep the comparison honest:

- **Fix the rule before you look at results.** The strata may use the measurements but not
  the outcomes, the signal or anything a fitted model produced. Cut points chosen after
  seeing which features were selected can be tuned, knowingly or not, to flatter a method.
- **Check that chance has room to vary.** `admissible_panel_count` is the number of ways the
  random reference can rearrange the selection within its strata: 6 here, the ways to choose
  two of four paired features. If it is 1, the only possible random selection is the
  observed one and recovery above chance is zero whatever the model did. Use fewer, larger
  strata.
- **Prepare the matrix first.** `np.corrcoef` returns missing values for a column that is
  constant or contains missing values. Drop constant columns and compute correlations on
  complete rows or with pairwise deletion (for example `pandas.DataFrame.corr`).

## Change the signal structure

Use the same measurements to compare a sparse signal from A and B with a signal from the
mean of each pair's standardised measurements. `materialize_truth` computes the generating
score and recovery targets without simulating outcomes. To simulate outcomes, pass the signal
specification, such as `SparseTruth(features=("A", "B"))`, to `generate` or `generate_partitioned`.

```python
sparse = ppm.materialize_truth(ppm.SparseTruth(features=("A", "B")), population, seed=3)
distributed = ppm.materialize_truth(
    ppm.CorrelatedTruth(group_names=("A_group", "B_group"), signal_form="group_means"),
    population,
    feature_sets=groups,
    seed=3,
)
print(sparse.direct_features)  # ('A', 'B')
print(distributed.direct_features)  # ()
```

Group means leave `direct_features` empty. Use `evaluate_groups` for the fraction of groups
with a member selected. Each member contributes to its group mean, so `evaluate_module` reports the
fraction of total absolute generating weight selected. Repeat the comparison on independently
simulated datasets, including a null signal and stronger effects. Report prediction, precision
and recovery across replicates. For continuous responses, counts, ordered categories and
time-to-event data, see [Outcomes](outcomes.md).

## Choose discrimination and compare refits

- Set `BinaryOutcome(probability=0.15, target_auc=0.7)` so generation solves for the odds ratio
  that gives the generating score an expected AUC of 0.7. [AUC calibration](calibration.md#what-is-solved)
- Measure selection agreement across refits or combine their importance scores into one ranking.
  [`selection_stability`](metrics.md#selection-stability) and
  [`consensus_ranking`](metrics.md#consensus-and-rank-biased-overlap)
- Compare two fixed fitted predictors by resampling their shared evaluation observations.
  [`paired_bootstrap_difference`](metrics.md#paired-bootstrap-and-discrimination-gap)
- Compare two procedures by running both on each independently simulated dataset.
  [`compare_paired_replicates`](evaluation.md#paired-procedure-comparisons)
