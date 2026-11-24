---
name: pyplasmode
description: Design, run and interpret plasmode simulations with the PyPlasmode Python library. It generates outcomes from features you choose inside real or synthetic tabular data, so you can measure whether a model and its feature-importance method recover those features, as well as how well the model predicts. Use this skill whenever the user wants to test, validate or benchmark feature selection, biomarker discovery, variable importance, SHAP values or any other attribution against a known answer; asks whether the top-ranked features are the real ones; wants a simulation study that keeps the correlations of their own data; asks how many samples or events are needed before a selection can be trusted; or mentions pyplasmode, plasmode simulation, generating features, recovery or recovery above chance, even if they do not name the library.
---

# PyPlasmode

PyPlasmode answers one question that prediction metrics cannot: when a model ranks features, are the top ones the features that produced the outcome? It keeps the user's feature matrix as it is, generates a new outcome from features you choose, and then scores a ranking against that known answer. Model fitting, tuning and explanation stay in the user's own tools.

Use it to check a method before its selections are relied on, to compare methods, or to find out how large a study must be before a selection can be trusted. Do not use it to fit models or to claim that a recovered feature causes anything in the real data: recovery is measured against a simulated outcome.

## Terms

Use these words with the user, and give the code name once when it differs.

| Term | Meaning | In code |
|:--|:--|:--|
| generating features | the features used to generate the outcome | `truth.direct_features` |
| generating score | the number computed for each row from them | `truth.signal` |
| signal structure | sparse, correlated, distributed, interaction, null or custom | `SparseTruth`, `CorrelatedTruth`, `ModuleTruth`, `InteractionTruth`, `NullTruth`, `CustomTruth` |
| correlates | measured features strongly correlated with a generating feature | other members of its `FeatureSet` |
| selection | the features a method picks | a ranking, or `nomination_from_scores(...)` |
| recovery | fraction of the generating features (or groups, or weight) in a selection | `recall`, `observed_recovery` |
| recovery above chance | recovery minus what random lists of the same size and composition recover | `chance_adjusted_recovery` |
| AUC of the generating score | the best discrimination attainable for that outcome | `oracle_auc` |

## Workflow

1. Wrap the data: a numeric matrix with one row per observation and one named column per feature. Missing values may stay.
2. Split the source rows before generating anything, so no row is in both training and test.
3. Choose a signal structure and an outcome in observable units (a probability, a mean, a rate, an event risk).
4. Generate training, validation and test samples.
5. Fit the user's model and compute importance exactly as they would on real data.
6. Score prediction on the test sample and recovery against the known generating features.
7. Repeat over independent seeds and report the spread. One simulated dataset is an anecdote.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pyplasmode as ppm

# 1. Rows are observations; columns are named numeric features.
rng = np.random.default_rng(0)
names = tuple(f"f{i}" for i in range(50))
population = ppm.Population(rng.normal(size=(3_000, 50)), names)

# 2. Split source rows first.
partition = ppm.partition_population(
    population, training_fraction=0.6, validation_fraction=0.2, seed=1
)

# 3 and 4. Generate an outcome from chosen features.
samples = ppm.generate_partitioned(
    partition,
    truth=ppm.SparseTruth(features=("f3", "f17", "f42")),
    outcome=ppm.BinaryOutcome(probability=0.15, odds_ratio=2.0),
    sample_sizes=ppm.PartitionSampleSizes(1_800, 600, 600),
    sampling_method="without_replacement",
    seed=2,
)
train, test = samples.training, samples.evaluation

# 5. The user's model and importance measure (here, standardised coefficients).
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000))
model.fit(train.X, train.outcome.values)
scores = np.abs(model[-1].coef_[0])

# 6. Prediction on held-out rows, and recovery of the generating features.
auc = ppm.evaluate_prediction(
    test.outcome.values, model.predict_proba(test.X)[:, 1], metric="roc_auc"
).value
selection = ppm.nomination_from_scores(train.feature_ids, scores, depth=10)
matched = ppm.evaluate_matched_recovery(
    selection, train.truth, estimand="exact", strata=("all",) * len(names)
)
print(round(auc, 3), matched.observed_recovery, matched.chance_adjusted_recovery)
```

Importance scores passed to `nomination_from_scores` must be non-negative and in the same feature order as the data; take absolute values of signed coefficients first. Scores of zero are never selected, so a sparse model can return fewer features than the depth asked for.

## Choosing the signal structure

Pick the structure that matches what the user believes about their problem, and when they have no belief, run several: methods differ most between structures.

| Structure | Use when | Score recovery with |
|:--|:--|:--|
| `SparseTruth(features=...)` or `SparseTruth(count=k)` | a few features act independently | `estimand="exact"` |
| `CorrelatedTruth(group_names=..., sentinels=...)` | each generating feature has strong measured correlates | `"exact"` for the feature itself, `"group"` to credit any member of its group |
| `ModuleTruth(set_name=...)` | many features of one set each contribute a little | `"module"` |
| `InteractionTruth(features=(a, b))` | the outcome depends on two features jointly (product or joint threshold) | `"interaction"` |
| `NullTruth()` | checking what a method selects when nothing is there | count and recurrence of selections |
| `CustomTruth(function)` | anything else, for example a saturating effect | the targets the function returns |

Correlated and distributed structures need named groups passed as `feature_sets=ppm.FeatureSets((ppm.FeatureSet(name, members), ...))`. The user supplies the groups, for example from a correlation clustering or pathway annotation computed without looking at any outcome.

## Choosing the outcome

State outcomes in the units the user thinks in; the library solves for the model parameters.

- `BinaryOutcome(probability=0.15, odds_ratio=2.0)`: the odds ratio is per standard deviation of the generating score, not per feature.
- `TimeToEventOutcome({ppm.years(10): 0.05}, hazard_ratio=2.0, censoring_risks={ppm.years(10): 0.05})`: cumulative event risks at chosen times; returns `outcome.time` in days and `outcome.event`.
- `ContinuousOutcome`, `CountOutcome`, `OrdinalOutcome` for the other outcome types.
- To fix difficulty instead of effect size, give `target_auc=0.7` in place of the ratio (binary and time-to-event). The solved ratio is in `outcome.calibration["odds_ratio"]` or `["hazard_ratio"]`.

Match the event rate, sample size and plausible effect size of the user's real problem. A simulation at ten times the real event rate says little about the real study.

## Reading the results

- **Prediction and recovery are different properties.** A model can predict well from correlates while selecting none of the generating features. Report both, never one as a stand-in for the other.
- **Compare recovery with chance.** Ten features drawn at random from 100 recover 10% of anything. Use `evaluate_matched_recovery` and report recovery above chance. Its `strata` argument groups features that are equally easy to pick by accident (for example by the size of their correlation group); define strata from the measurements alone, before any outcome exists.
- **Compare with the attainable.** The AUC of the generating score is the ceiling for that outcome. `pyplasmode.metrics.discrimination_gap` reports how far a fitted model falls short.
- **Recovery has its own sample size.** It can keep rising with more events after the AUC has levelled off. If recovery is low, rerun at a higher event rate or sample size before concluding that the method cannot identify the features (recipe 3).
- **Check what agreement between methods is worth.** Run two or three methods that model the outcome differently and compare how often a feature selected by all of them is a generating feature with how often one selected by a single method is.
- **A stable list is not a correct list.** Check that a method's selections change when the outcome changes: compare its lists across unrelated simulated outcomes, and include a null outcome.

## What goes wrong

- **Generating on the full data and splitting afterwards** lets resampled copies of one row land in training and test. Partition first, then `generate_partitioned`.
- **Rows that belong together** (repeat visits, relatives) must be split as a unit by the user before generation; the library splits rows.
- **Feature order.** Recovery functions expect scores and rankings in the order of `feature_ids`. Build rankings from `train.feature_ids`, not from a re-sorted data frame.
- **A constant or all-missing column** cannot be a generating feature and raises `ValueError` if chosen.
- **Rare draws fail on purpose.** `RealizationError` means the sampled outcome landed far from its target; count such failures in the summary and keep going with the next seed. `CalibrationError` means the requested target cannot be reached and the specification must change.
- **Time-to-event evaluation** needs follow-up beyond the horizon. To evaluate at ten years, either add a later risk point or set `administrative_horizon=ppm.years(11)`, and pass times in days: `horizon=ppm.years(10).days`.
- **Privacy.** Generated datasets contain the source rows. Treat them with the access conditions of the source data and do not write them to shared locations.

## Reporting

Give the user, for every setting studied: the signal structure, effect size, sample size and number of events; the number of simulated datasets and how many failed; prediction on held-out rows with the AUC of the generating score beside it; recovery and recovery above chance with an interval across simulated datasets (`ppm.summarize_replicates`); and, when methods are compared, paired differences on the same simulated datasets (`ppm.compare_paired_replicates`). Say plainly that recovery was measured on simulated outcomes placed in measured features, which is the favourable case.

## Going further

- `references/recipes.md` has complete, tested recipes: repeating a study and comparing two methods, correlated groups with group recovery, sizing a study by event rate, time-to-event outcomes with censoring, and setting difficulty by target AUC. Read it before writing any of these from scratch.
- Definitions of every metric and outcome family are in the documentation: https://bradsegal.github.io/PyPlasmode/ (Methods, Generating signals, Outcomes, Evaluation, Metrics, AUC calibration).
