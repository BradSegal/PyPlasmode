# PyPlasmode recipes

Complete examples for common studies. Each runs as written on synthetic data with the core dependencies (NumPy, SciPy, scikit-learn). For empirical data, substitute the user's matrix and feature IDs, choose valid generating features and partition sizes, and adapt the model to handle the returned missing values without learning any preprocessing from test rows.

Contents:

1. Repeat a study and compare two procedures
2. Correlated features and group recovery
3. How recovery varies with sample size
4. Time-to-event outcomes with censoring
5. Set the difficulty with a target AUC

## 1. Repeat a study and compare two procedures

One simulated dataset cannot separate a procedure's behaviour from luck. This five-draw example compares two fixed procedures on the same generated datasets, for one synthetic source matrix and one partition. For a study, choose enough draws for the Monte Carlo precision you need, and vary the source matrix or the partition only if they are part of the question.

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


ids, linear, forest = [], [], []
generation_failures = []
fit_failures = {"linear": [], "forest": []}
for seed in range(5):
    ids.append(f"dataset_{seed}")
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
        generation_failures.append(ids[-1])
        linear.append(None)  # the draw stays in the denominator
        forest.append(None)
        continue
    train = samples.training
    for name, score_fn, results in (
        ("linear", linear_scores, linear),
        ("forest", forest_scores, forest),
    ):
        if name == "linear" and np.unique(train.outcome.values).size < 2:
            fit_failures[name].append((ids[-1], "one training class"))
            results.append(None)
            continue
        scores = score_fn(train)
        selection = ppm.nomination_from_scores(train.feature_ids, scores, depth=8)
        matched = ppm.evaluate_matched_recovery(
            selection, train.truth, estimand="exact", strata=strata
        )
        results.append(matched.chance_adjusted_recovery)

print("attempted", len(ids), "generation failures", generation_failures)
for name, results in (("linear", linear), ("forest", forest)):
    succeeded = [value for value in results if value is not None]
    print(name, "successful", len(succeeded), "fit failures", fit_failures[name])
    if len(succeeded) >= 2:
        print(ppm.summarize_replicates(succeeded, failure_count=len(ids) - len(succeeded)).mean)
    else:
        print("mean", succeeded[0] if succeeded else "undefined", "interval undefined")

complete = sum(a is not None and b is not None for a, b in zip(linear, forest, strict=True))
if complete >= 2:
    comparison = ppm.compare_paired_replicates(tuple(ids), linear, tuple(ids), forest)
    print(comparison.summary.mean)  # forest minus linear, on the datasets where both ran
else:
    print("paired difference undefined: fewer than two complete pairs")
```

`SparseTruth(count=4)` uses a seeded search for four features under the pairwise correlation threshold, so the generating features can differ between seeds; name the features explicitly to study one particular set. Report each procedure's mean, the paired difference with its Monte Carlo interval, the attempted and successful counts and the causes of failure. The paired estimate is conditional on both procedures succeeding. When adapting this recipe, decide in advance which fitting or evaluation failures make a procedure's estimate unavailable, record the procedure, draw and reason, and insert `None` for that procedure; investigate specification errors and unexpected exceptions separately.

## 2. Correlated features and group recovery

When a generating feature has close correlates, a method may select a correlate instead. Exact recovery asks for the generating feature itself; group recovery credits any member of its group. Report both. This single fit illustrates the scoring rules; repeat it (recipe 1) to compare procedures or estimate performance.

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

Here the chance reference keeps the number of positive scores and the selection weights within the paired and the single columns. Choose strata for the application, and check that features are plausibly exchangeable within them. Group recovery can have a higher chance expectation than exact recovery, because a group offers several ways to be hit.

## 3. How recovery varies with sample size

Recovery can keep improving after prediction has levelled off. To see how it depends on the size of a study, hold the generating features, the outcome targets and the procedure fixed and vary the training size. This design recalibrates the score and the outcome on the combined sampled rows at each size; to isolate training size under one fixed generating mechanism, fix its transformation and outcome parameters across sizes.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

import pyplasmode as ppm

rng = np.random.default_rng(7)
names = tuple(f"f{i}" for i in range(60))
population = ppm.Population(rng.normal(size=(6_000, 60)), names)
partition = ppm.partition_population(
    population, training_fraction=0.6, validation_fraction=0.2, seed=1
)
truth = ppm.SparseTruth(features=("f1", "f2", "f3", "f4", "f5", "f6"))
for training_size in (400, 3_600):
    samples = ppm.generate_partitioned(
        partition,
        truth=truth,
        outcome=ppm.BinaryOutcome(probability=0.1, odds_ratio=2.0),
        sample_sizes=ppm.PartitionSampleSizes(training_size, 400, 1_200),
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
    print(training_size, events, round(auc, 3), round(matched.chance_adjusted_recovery, 3))
```

The two draws illustrate the comparison; estimate a trend or a threshold from repeated draws at each size. For sizing, vary training size at the intended event probability, keep the evaluation sample adequate, and fix in advance the selection rule, the acceptable recovery and precision, and the Monte Carlo precision. Repeat each setting (recipe 1), report the realised event counts with their uncertainty, and identify the smallest size studied that meets the criteria under each plausible design. Treat event counts as specific to the scenario; prevalence, effect size and signal structure need their own sensitivity runs.

## 4. Time-to-event outcomes with censoring

Give cumulative event risks before censoring and, optionally, independent censoring risks at the same time points. Estimate discrimination at a horizon with `cumulative_dynamic_auc`, which weights by the censoring distribution of the training sample. The example adds a later risk point to extend follow-up beyond the horizon, which also specifies event and censoring risk up to that point.

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
# The generating score, as the benchmark for this hazard ratio above one. Replace it with
# the user's survival model: any score where higher means higher risk.
risk = test.truth.signal
auc = cumulative_dynamic_auc(
    train.outcome.time,
    train.outcome.event,
    test.outcome.time,
    test.outcome.event,
    risk,
    horizon=horizon.days,
)
cases = test.outcome.event & (test.outcome.time <= horizon.days)
censored = ~test.outcome.event & (test.outcome.time <= horizon.days)
print(round(auc, 3), int(cases.sum()), int(censored.sum()))  # AUC, cases, censored by ten years
```

Fit the survival model on the training features with `train.outcome.time` (in days) and `train.outcome.event`, and pass a test score in which higher means higher risk. For a row censored at or before the horizon without an event, the status at the horizon is unknown. Decide in advance how a classifier of "event by the horizon" handles those unknown labels, with a censoring-aware approach such as inverse-probability weighting where appropriate, and evaluate both scores on the same people, horizon and metric.

## 5. Set the difficulty with a target AUC

When the user thinks in terms of how predictable the outcome is, specify the expected AUC of the generating model and let the library solve the effect size.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pyplasmode as ppm
from pyplasmode.metrics import discrimination_gap

rng = np.random.default_rng(12)
names = tuple(f"f{i}" for i in range(30))
population = ppm.Population(rng.normal(size=(6_000, 30)), names)
partition = ppm.partition_population(
    population, training_fraction=0.5, validation_fraction=0.1, seed=2
)
samples = ppm.generate_partitioned(
    partition,
    truth=ppm.SparseTruth(features=("f0", "f1", "f2")),
    outcome=ppm.BinaryOutcome(probability=0.1, target_auc=0.75),
    sample_sizes=ppm.PartitionSampleSizes(3_000, 600, 2_400),
    sampling_method="without_replacement",
    seed=3,
)
train, test = samples.training, samples.evaluation
calibration = test.outcome.calibration
print(round(calibration.oracle_auc, 3), round(calibration["odds_ratio"], 2))

model = LogisticRegression(max_iter=1_000).fit(train.X, train.outcome.values)
gap = discrimination_gap(
    test.truth.signal,
    model.predict_proba(test.X)[:, 1],
    metric=lambda scores: roc_auc_score(test.outcome.values, scores),
)
print(round(gap.gap, 3))  # AUC of the generating score on held-out rows minus the fitted AUC
```

The target is an expectation over the combined rows used for calibration. On held-out rows the realised AUC varies, and in a finite sample either score can have the higher AUC.
