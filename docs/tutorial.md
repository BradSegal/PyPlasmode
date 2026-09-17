# Benchmarking feature selection

Suppose a model predicts well and ranks a few features highly. Are those the features
that generated the outcome, or alternatives carrying similar information? This tutorial
builds a dataset where the answer is known, fits a logistic model and measures both its
prediction accuracy and its feature recovery.

The measurements are synthetic. For your own study, supply a numeric matrix with one
observation per row and one feature per column, together with unique column names.
Those features can be biomarkers or other measured variables. Install
[PyPlasmode](https://pypi.org/project/pyplasmode/); NumPy and scikit-learn are included.
Run the following blocks in order.

## Create correlated measurements

Two feature pairs share information; four other features vary independently. These
relationships are defined before any outcome is generated.

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

## Generate an outcome from known features

Only A and B enter the generating score. A_proxy and B_proxy are correlated alternatives:
they can help predict the outcome without contributing to its construction. The API calls
the generating member of each group a *sentinel*. Here, the outcome has a 15% marginal
probability, and its odds double for each standard-deviation increase in the combined signal.

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
rows. The test sample is used only for evaluation. The separate validation sample is unused
here because tuning uses cross-validation.

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

The prediction result reports test-set ROC AUC, which measures how well the model ranks
positive outcomes above negative ones. The recovery results answer a different question:
did the highest-ranked features include A and B? Recall is the fraction of generating
features found; precision is the fraction of selected features that generated the signal.
Selecting A and B gives both scores 1.0. Selecting A and B_proxy gives both scores 0.5,
but still recovers both groups.

## Compare with matched chance

A random list can recover a generating feature by chance. To make a fair comparison,
keep the number of selections from correlated pairs and independent features the same.
These two categories form the *strata* for the random reference. Within each stratum,
feature identities are treated as exchangeable.

```python
nomination = ppm.nomination_from_scores(names, scores, depth=2)
strata = ("pair", "pair", "pair", "pair", "singleton", "singleton", "singleton", "singleton")
matched = ppm.evaluate_matched_recovery(nomination, train.truth, estimand="exact", strata=strata)
print(matched.observed_recovery, matched.expected_recovery, matched.chance_adjusted_recovery)
```

If both selections come from the four paired features, random assignment recovers half
the two generating features on average. Finding both gives recovery 1.0, expected recovery
0.5 and an excess of 0.5. Finding just A and B_proxy gives exact recovery 0.5 and zero
excess under this reference. This asks whether the method identifies A and B better than
random selection within the same types of features.

## Change the generating mechanism

The same measurements can test an outcome driven by individual features or by a shared
group signal. A sparse specification uses A and B directly. A group-mean specification
uses both members of each pair, so recovery concerns the group and its combined weight.

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

An empty `direct_features` tuple in the group-mean case denotes a distributed recovery
target. Use `evaluate_groups` or `evaluate_module` for that hypothesis. Repeat the
comparison over independent outcome draws, include a null signal and stronger effects,
and report the resulting distribution of prediction, precision and recovery. The
[outcome examples](outcomes.md) extend this design to continuous responses, counts,
ordered categories and time-to-event data.
