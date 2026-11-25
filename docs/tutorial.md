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

Recovery above chance subtracts the recovery expected from random lists of the same size
and composition. Here, the random lists match the number of selections (`nomination` in the API)
from correlated pairs and independent features. These two categories form the *strata* for the random reference. Within each
stratum, feature identities are treated as exchangeable.

```python
nomination = ppm.nomination_from_scores(names, scores, depth=2)
strata = ("pair", "pair", "pair", "pair", "singleton", "singleton", "singleton", "singleton")
matched = ppm.evaluate_matched_recovery(nomination, train.truth, estimand="exact", strata=strata)
print(matched.observed_recovery, matched.expected_recovery, matched.chance_adjusted_recovery)
```

If both selections come from the four paired features, random assignment recovers half
the two generating features on average. Finding both gives recovery 1.0, expected recovery
0.5 and recovery above chance 0.5 (`chance_adjusted_recovery`). Finding just A and B_proxy
gives recovery 0.5 and recovery above chance zero.

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

## Matching strata on empirical measurements

For an empirical matrix, define strata before generating outcomes or fitting
models. For example, bin each feature by its strongest absolute correlation
with another feature. Estimate these correlations on the source or training
measurements, keeping evaluation outcomes out of this step:

```python
correlations = np.abs(np.corrcoef(population.X, rowvar=False))
np.fill_diagonal(correlations, 0.0)
max_correlation = correlations.max(axis=0)
correlation_bins = np.digitize(max_correlation, [0.3, 0.7])
empirical_strata = tuple(f"correlation_{int(b)}" for b in correlation_bins)
assert len(empirical_strata) == len(population.feature_ids)
```

Compute correlations on complete, non-constant features. Resolve missingness or
use missingness-rate bins before computing correlations on other matrices.
Choose bin boundaries before examining recovery. Check that the matched random selection permits
more than one assignment across the strata (`admissible_panel_count > 1`).
