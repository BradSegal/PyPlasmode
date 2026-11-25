---
name: pyplasmode
description: Design, run and interpret plasmode simulations with the PyPlasmode Python library. Generate outcomes under a known signal using empirical or synthetic tabular features, then measure prediction and recovery of the specified targets. Use this skill when a user wants to benchmark feature selection, biomarker discovery, variable importance, SHAP values or any other attribution against known targets; asks whether top-ranked features are the real ones; wants a simulation that draws complete rows from their own feature data; wants to estimate how recovery changes with sample size and event probability under stated scenarios; or mentions pyplasmode, plasmode simulation, generating features, recovery or recovery above chance, even if they do not name the library.
---

# PyPlasmode

PyPlasmode answers a question that prediction metrics cannot: when a procedure ranks features, are the top ones the features that produced the outcome? It samples complete rows from the source feature matrix, keeping their values and missingness, generates an outcome from a signal you specify, and scores selections against the known targets. Model fitting, tuning and explanation stay in the user's own tools.

Use it to compare procedures, or to estimate the sample size that meets a prespecified performance criterion under stated simulation assumptions. Do not use it to fit models, and do not claim that a recovered feature causes anything in the real data: recovery is measured against a simulated outcome.

## Terms

Use these words with the user, and give the code name once when it differs.

| Term | Meaning | In code |
|:--|:--|:--|
| generating features | features used to construct the generating score | `truth.direct_features` for exact targets; `truth.weights` records contributions for weighted structures |
| generating score | the number computed for each row from them | `truth.signal` |
| signal structure | sparse, correlated, distributed, interaction, null or custom | `SparseTruth`, `CorrelatedTruth`, `ModuleTruth`, `InteractionTruth`, `NullTruth`, `CustomTruth` |
| correlates | measured features strongly correlated with a generating feature | other members of its `FeatureSet` in a sentinel design; verify the correlations on the features |
| selection | the features nominated by a procedure | leading entries of a ranking, or memberships from `nomination_from_scores(...)` |
| recovery | the proportion of exact features, groups or generating weight recovered, or the probability of selecting both members of an interaction | `recall` for exact recovery from a ranking; `observed_recovery` for matched recovery |
| recovery above chance | observed recovery minus its expected value under random assignment within matched strata | `chance_adjusted_recovery` |
| expected AUC of the generating model | discrimination of the model-implied event risk on the calibration rows, at the specified horizon for survival | `outcome.calibration.oracle_auc` when defined |

## Workflow

1. Wrap a nonempty numeric matrix with a unique feature ID per column. Missing values can stay in the source matrix, but the fitted model must then handle them.
2. Split the source sampling units before resampling. Keep each person or cluster within one role when observations are related.
3. Choose a signal structure and an outcome in observable units (a probability, a mean, a rate, an event risk).
4. With a built-in signal, generate training, validation and evaluation samples with `generate_partitioned`. For built-in signals, the roles share the imputation and scaling of the generating features, the scaling of the generating score and the outcome calibration, all computed on their combined sampled rows. A target therefore holds over the combined rows; expectations and realised outcomes in a single role can differ. For transfer under a fixed generating mechanism, specify the transformation and the outcome parameters separately and carry them across populations.
5. Fit the model and its preprocessing on training data. Use validation data or training-only resampling for tuning and for choosing a model, a feature transformation, a selection depth or an importance procedure. Keep test outcomes out of those choices.
6. Evaluate the prespecified prediction metrics on the test sample. Apply the chosen procedure and score its selections against the recovery targets of the signal structure.
7. Repeat over independent draws for the declared design, keeping methods paired on the same draw. State which features, partitions and generating features are fixed and which change, and choose the number of draws for the Monte Carlo precision you need. One simulated dataset is an anecdote.

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

Supply one finite, non-negative importance score per original feature, using absolute values of signed coefficients when the target is the size of an effect. The depth is an integer from one to the number of features. Scores of zero get no membership, and positive scores tied at the cutoff share the remaining places, so report the total membership and the number of features with nonzero membership.

## Choosing the signal structure

Choose signal structures that represent plausible mechanisms. Include several when the mechanism is uncertain, and report performance for each.

| Structure | Use when | Score recovery with |
|:--|:--|:--|
| `SparseTruth(features=...)` or `SparseTruth(count=k)` | a few features enter the score additively | `estimand="exact"` |
| `CorrelatedTruth(group_names=..., sentinels=...)` | each generating feature has strong measured correlates | `"exact"` for the feature itself, `"group"` to credit any member of its group |
| `CorrelatedTruth(group_names=..., signal_form="group_means")` | within-group averages enter the score | `"group"` or weighted `"module"`; there are no exact targets |
| `ModuleTruth(set_name=...)` | many features of one set each contribute a little | `"module"` |
| `InteractionTruth(features=(a, b))` | the outcome depends on two features jointly (product or joint threshold) | `"interaction"` |
| `NullTruth()` | checking selection when no feature drives the outcome | count false selections and how often they recur; recovery is undefined |
| `CustomTruth(function)` | a deterministic function returning a `CustomSignal` with one finite, nonconstant score per sampled row and explicit recovery targets | the evaluator that matches the declared target |

Correlated and distributed structures need named groups passed as `feature_sets=ppm.FeatureSets((ppm.FeatureSet(name, members), ...))`. Use feature IDs present in the matrix. Correlated groups must be disjoint, and each named sentinel must belong to its group. Define the groups from measurements or annotation, independently of any generated outcome, and check that groups meant to hold correlates have the correlations you intend. Custom signals work with `generate`; for a held-out design with a custom signal, implement and verify one generating mechanism over disjoint source partitions.

## Choosing the outcome

State outcomes in the units the user thinks in; the library solves for the model parameters.

- `BinaryOutcome(probability=0.15, odds_ratio=2.0)`: the odds ratio is per standard deviation of the generating score, not per feature.
- `TimeToEventOutcome({ppm.years(10): 0.05}, hazard_ratio=2.0, censoring_risks={ppm.years(10): 0.05})` specifies the ten-year event risk before censoring and an independent censoring risk at the same time point. `outcome.time` is observed follow-up in days and `outcome.event` marks an observed event.
- `ContinuousOutcome`, `CountOutcome`, `OrdinalOutcome` for the other outcome types.
- For binary or time-to-event outcomes, `target_auc=0.7` can be the sole effect specification; the library then solves the odds or hazard ratio per standard deviation of the generating score. For survival, set an `auc_horizon` before administrative follow-up ends. The solved ratio is in `outcome.calibration["odds_ratio"]` or `["hazard_ratio"]`.

Match sample size, outcome frequency, plausible effect sizes and measurement conditions to the application. Add sensitivity scenarios for signal structure, missingness and dependence between features where they could change the conclusion. The results are conditional on those assumptions.

## Reading the results

- **Prediction and recovery are different properties.** A model can predict well from correlates while selecting none of the generating features. Report both, never one as a stand-in for the other.
- **Compare recovery with chance.** A random list of ten among 100 exchangeable features recovers 10% of any nonempty exact target set on average. Define strata in advance from properties of the features, independently of outcomes and of the generating truth. `evaluate_matched_recovery` keeps the number of positive scores and the selection weights within each stratum. Report observed recovery, expected recovery and their difference. The difference depends on the matching design; it is not a significance test or an estimate of false discovery.
- **Compare with the generating model.** Report the expected AUC from calibration separately. Use `pyplasmode.metrics.discrimination_gap` to subtract the fitted AUC from the AUC of the model-implied event risk on the same test outcomes, with the same metric and, for survival, the same horizon and censoring reference. The generating score multiplied by the calibrated `signal_coefficient` gives that risk ordering; for an odds or hazard ratio below one the raw score points the wrong way. In a finite sample the gap can be negative.
- **Recovery can improve after prediction levels off.** Vary training size at the application's event probability, and use separate sensitivity scenarios for prevalence, effect size and signal structure (recipe 3).
- **Measure what agreement between methods is worth.** Fix the data and the selection depths in advance, then compare the fraction of generating features among selections made by every method with the fraction among features selected by exactly one. Report both denominators and the uncertainty across datasets.
- **Check that selections depend on the outcome.** Use `evaluate_outcome_specificity` to compare the same selection's recovery above chance for the generating targets and for prespecified, non-overlapping targets of the same kind and count. Under a null outcome, report false selections and how often they recur against a reference chosen in advance. A rule that always returns the top k fills its places even under the null.

## What goes wrong

- **Partition before resampling.** Resampling a pooled source with replacement and splitting afterwards can send copies of one record to both training and test. Use `generate_partitioned` with a source partition, and split related observations at the person or cluster level before generation. The built-in sampler draws individual rows; a study that resamples whole clusters needs its own cluster-level resampling and a generating mechanism suited to the dependence within clusters.
- **Feature order.** Align the scores and strata for a selection to its `feature_ids`. Map transformed columns back to the original features when the recovery target is defined there. `evaluate_ranking` needs a complete ordering of the feature IDs, most important first.
- **Columns without variation.** A generating column in a built-in structure must vary; one that is constant or entirely missing raises `ValueError`. A custom function handles its own inputs and must return a finite, nonconstant score.
- **Count failures at each step.** `RealizationError` means a generated outcome landed outside its diagnostic tolerance: keep the attempted seed in the denominator, record the failure and go on to the next planned draw. `CalibrationError` means the target or the specification needs inspecting. Generation can also succeed with too few events or classes for a method to fit or be evaluated; record those failures by method, and read summaries as conditional on the draws that succeeded.
- **Time-to-event evaluation.** For a ten-year AUC, follow-up must extend beyond ten years, through a later risk point or `administrative_horizon=ppm.years(11)`. A later administrative cutoff alone adds observation time without adding events or censoring after the last risk point. Pass `horizon=ppm.years(10).days`. Check that observed training follow-up extends beyond the horizon, that the training censoring survival is positive at the horizon and at the test case times, and that the test sample has observed cases and controls at the horizon.
- **Privacy.** Generated matrices contain source records. Apply the source data's access and sharing conditions to the matrices, outcomes and source-row indices, and store them only in approved locations.

## Reporting

For every setting, report the sampling unit, the signal structure and effect, the outcome model, the training and evaluation sizes, and the event or class counts. Report attempted and successful draws, the causes of failure, and prediction metrics on held-out rows. For binary or survival AUC, distinguish the expected AUC of the generating model from the AUC of the model-implied event risk realised on the same evaluation rows as the fitted model. Report observed recovery, expected recovery and recovery above chance, and for exact targets the precision at the attained membership (recovered target membership divided by the total membership, when that is positive). Use `ppm.summarize_replicates` for the mean and its Monte Carlo interval over at least two successful draws, and `ppm.compare_paired_replicates` to compare procedures on shared datasets, saying when a result is conditional on both succeeding. State that the conclusions concern simulated outcomes under the specified signal and feature distribution.

## Going further

- `references/recipes.md` has complete, tested recipes: repeating a study and comparing two procedures, correlated groups with group recovery, how recovery varies with sample size, time-to-event outcomes with censoring, and setting difficulty by target AUC. Read it before writing any of these from scratch.
- Definitions of every metric and outcome family are in the documentation: https://bradsegal.github.io/PyPlasmode/ (Methods, Generating signals, Outcomes, Evaluation, Metrics, AUC calibration).
