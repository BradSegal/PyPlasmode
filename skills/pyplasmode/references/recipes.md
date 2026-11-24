# PyPlasmode recipes

Complete examples for common studies. Each runs as written on synthetic data with the core dependencies (NumPy, SciPy, scikit-learn). Replace the synthetic matrix with the user's own numeric matrix and feature names, and the model with theirs.

Contents:

1. Repeat a study and compare two methods
2. Correlated features and group recovery
3. How many events are needed
4. Time-to-event outcomes with censoring
5. Set the difficulty with a target AUC

## 1. Repeat a study and compare two methods

One simulated dataset cannot separate a method's behaviour from luck. Generate several from independent seeds, run both methods on each, and compare them pairwise on the same datasets.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pyplasmode as ppm

rng = np.random.default_rng(0)
names = tuple(f"f{i}" for i in range(40))
population = ppm.Population(rng.normal(size=(2_000, 40)), names)
partition = ppm.partition_population(
    population, training_fraction=0.6, validation_fraction=0.2, seed=1
)
strata = ("all",) * len(names)


def linear_scores(train):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000))
    model.fit(train.X, train.outcome.values)
    return np.abs(model[-1].coef_[0])


def forest_scores(train):
    model = RandomForestClassifier(n_estimators=50, random_state=0)
    model.fit(train.X, train.outcome.values)
    return model.feature_importances_


ids, linear, forest, failures = [], [], [], 0
for seed in range(5):
    try:
        samples = ppm.generate_partitioned(
            partition,
            truth=ppm.SparseTruth(count=4),
            outcome=ppm.BinaryOutcome(probability=0.2, odds_ratio=2.0),
            sample_sizes=ppm.PartitionSampleSizes(1_200, 400, 400),
            sampling_method="without_replacement",
            seed=seed,
        )
    except ppm.RealizationError:
        failures += 1
        continue
    train = samples.training
    ids.append(f"dataset_{seed}")
    for scores, results in ((linear_scores(train), linear), (forest_scores(train), forest)):
        selection = ppm.nomination_from_scores(train.feature_ids, scores, depth=8)
        matched = ppm.evaluate_matched_recovery(
            selection, train.truth, estimand="exact", strata=strata
        )
        results.append(matched.chance_adjusted_recovery)

print(ppm.summarize_replicates(linear, failure_count=failures).mean)
comparison = ppm.compare_paired_replicates(tuple(ids), linear, tuple(ids), forest)
print(comparison.summary.mean)  # forest minus linear, on the same simulated datasets
```

`SparseTruth(count=4)` draws different generating features for each seed, so the result does not depend on one lucky choice. Report the mean, its interval and the number of failed datasets.

## 2. Correlated features and group recovery

When a generating feature has close correlates, a method may select a correlate instead. Exact recovery asks for the generating feature itself; group recovery credits any member of its group. Report both, and build the chance reference from strata that separate grouped features from the rest.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

import pyplasmode as ppm

rng = np.random.default_rng(3)
X = rng.normal(size=(3_000, 12))
for leader, follower in ((0, 1), (2, 3), (4, 5)):
    X[:, follower] = 0.9 * X[:, leader] + np.sqrt(1 - 0.9**2) * X[:, follower]
names = tuple(f"f{i}" for i in range(12))
population = ppm.Population(X, names)
groups = ppm.FeatureSets(
    (
        ppm.FeatureSet("g0", ("f0", "f1")),
        ppm.FeatureSet("g1", ("f2", "f3")),
        ppm.FeatureSet("g2", ("f4", "f5")),
    )
)
plasmode = ppm.generate(
    population,
    truth=ppm.CorrelatedTruth(group_names=("g0", "g1", "g2"), sentinels=("f0", "f2", "f4")),
    feature_sets=groups,
    outcome=ppm.BinaryOutcome(probability=0.2, odds_ratio=2.5),
    sample_size=3_000,
    sampling_method="without_replacement",
    seed=5,
)
model = LogisticRegression(max_iter=1_000).fit(plasmode.X, plasmode.outcome.values)
selection = ppm.nomination_from_scores(names, np.abs(model.coef_[0]), depth=3)
strata = ("paired",) * 6 + ("single",) * 6
for estimand in ("exact", "group"):
    matched = ppm.evaluate_matched_recovery(
        selection, plasmode.truth, estimand=estimand, strata=strata
    )
    print(estimand, matched.observed_recovery, round(matched.chance_adjusted_recovery, 3))
```

A group is easier to hit by chance than a single feature, which is why the strata matter: the reference keeps the number of selections that fall among paired and single features.

## 3. How many events are needed

Recovery can keep improving long after the AUC has stopped. To size a study, hold the signal and the model fixed and vary the event rate or the sample size.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

import pyplasmode as ppm

rng = np.random.default_rng(7)
names = tuple(f"f{i}" for i in range(60))
population = ppm.Population(rng.normal(size=(4_000, 60)), names)
partition = ppm.partition_population(
    population, training_fraction=0.6, validation_fraction=0.2, seed=1
)
truth = ppm.SparseTruth(features=("f1", "f2", "f3", "f4", "f5", "f6"))
for probability in (0.02, 0.2):
    samples = ppm.generate_partitioned(
        partition,
        truth=truth,
        outcome=ppm.BinaryOutcome(probability=probability, odds_ratio=2.0),
        sample_sizes=ppm.PartitionSampleSizes(2_400, 800, 800),
        sampling_method="without_replacement",
        seed=11,
    )
    train, test = samples.training, samples.evaluation
    model = LogisticRegression(max_iter=1_000).fit(train.X, train.outcome.values)
    auc = ppm.evaluate_prediction(
        test.outcome.values, model.predict_proba(test.X)[:, 1], metric="roc_auc"
    ).value
    selection = ppm.nomination_from_scores(names, np.abs(model.coef_[0]), depth=6)
    matched = ppm.evaluate_matched_recovery(
        selection, train.truth, estimand="exact", strata=("all",) * len(names)
    )
    events = int(train.outcome.values.sum())
    print(events, round(auc, 3), round(matched.chance_adjusted_recovery, 3))
```

In a real study, repeat each setting over seeds (recipe 1) and read off the smallest number of events at which recovery above chance is acceptable for the user's purpose.

## 4. Time-to-event outcomes with censoring

Give cumulative event risks at the times that matter. Evaluate discrimination at a horizon with `cumulative_dynamic_auc`, which accounts for censoring. Follow-up must extend beyond the horizon, so add a later risk point.

```python
import numpy as np

import pyplasmode as ppm
from pyplasmode.metrics import cumulative_dynamic_auc

rng = np.random.default_rng(9)
names = tuple(f"f{i}" for i in range(20))
population = ppm.Population(rng.normal(size=(5_000, 20)), names)
partition = ppm.partition_population(
    population, training_fraction=0.6, validation_fraction=0.2, seed=1
)
horizon = ppm.years(10)
samples = ppm.generate_partitioned(
    partition,
    truth=ppm.SparseTruth(features=("f0", "f1")),
    outcome=ppm.TimeToEventOutcome(
        {horizon: 0.10, ppm.years(11): 0.11},
        hazard_ratio=2.0,
        censoring_risks={horizon: 0.05, ppm.years(11): 0.055},
    ),
    sample_sizes=ppm.PartitionSampleSizes(3_000, 1_000, 1_000),
    sampling_method="without_replacement",
    seed=4,
)
train, test = samples.training, samples.evaluation
# Stand-in for the user's survival model: any score where higher means higher risk.
risk = test.X[:, 0] + test.X[:, 1]
auc = cumulative_dynamic_auc(
    train.outcome.time,
    train.outcome.event,
    test.outcome.time,
    test.outcome.event,
    risk,
    horizon=horizon.days,
)
print(round(auc, 3), round(float(test.outcome.event.mean()), 3))
```

Fit the user's survival model on `train.outcome.time` and `train.outcome.event` (times are in days) and pass its risk score in place of `risk`. Classifiers that predict "event by the horizon" cannot use rows censored before the horizon without an event; say so when comparing them with survival models.

## 5. Set the difficulty with a target AUC

When the user thinks in terms of how predictable the outcome is, ask for the AUC of the generating score and let the library solve the effect size.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pyplasmode as ppm
from pyplasmode.metrics import discrimination_gap

rng = np.random.default_rng(12)
names = tuple(f"f{i}" for i in range(30))
population = ppm.Population(rng.normal(size=(6_000, 30)), names)
plasmode = ppm.generate(
    population,
    truth=ppm.SparseTruth(features=("f0", "f1", "f2")),
    outcome=ppm.BinaryOutcome(probability=0.1, target_auc=0.75),
    sample_size=6_000,
    sampling_method="without_replacement",
    seed=3,
)
calibration = plasmode.outcome.calibration
print(round(calibration.oracle_auc, 3), round(calibration["odds_ratio"], 2))

labels = plasmode.outcome.values
half = len(labels) // 2
model = LogisticRegression(max_iter=1_000).fit(plasmode.X[:half], labels[:half])
gap = discrimination_gap(
    plasmode.truth.signal[half:],
    model.predict_proba(plasmode.X[half:])[:, 1],
    metric=lambda scores: roc_auc_score(labels[half:], scores),
)
print(round(gap.gap, 3))  # AUC of the generating score minus the fitted AUC
```

The target is the expected AUC of the generating score over the supplied rows; the AUC realised in one sample varies around it, and a fitted model will usually fall below it.
