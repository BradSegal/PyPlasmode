# Benchmarking a protein ranking

Suppose you are choosing a method to identify proteins for biological follow-up. Its
predictions look useful, but you do not know whether the highest-ranked proteins are the
source of the signal or correlated alternatives. This tutorial creates an example in
which that distinction can be measured, then fits and explains a logistic model.

The measurements below are synthetic, so the example runs without participant data.
For your own study, replace the matrix and column names with your measured proteome.
Install `PyPlasmode`; NumPy and scikit-learn are included as dependencies. Run the
following blocks in order.

## Create correlated measurements

Two protein pairs share information; four other proteins vary independently. These
relationships are defined before any outcome is generated.

```python
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
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

## Generate an outcome with known contributors

Only A and B generate the outcome. Their proxies can carry information about it, but
do not enter the generating score. The API calls these contributing members *sentinels*.
A 15% event probability and an odds ratio of two per standard deviation of the combined
signal specify the outcome in interpretable units.

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

Model fitting remains in scikit-learn. Cross-validation selects regularisation within
training data, and the test participants are used only for evaluation. Absolute
standardised coefficients rank the proteins for this linear model; another model can
supply its own attribution scores to the same recovery functions.

```python
model = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(Cs=5, cv=5, scoring="roc_auc", max_iter=1_000),
)
model.fit(train.X, train.outcome.values)
prediction = ppm.evaluate_prediction(
    test.outcome.values, model.predict_proba(test.X)[:, 1], metric="roc_auc"
)
scores = np.abs(model[-1].coef_[0])
ranking = tuple(names[i] for i in np.argsort(-scores))
exact = ppm.evaluate_ranking(ranking, train.truth, depths=(2,)).at_depth[0]
group = ppm.evaluate_groups(ranking, train.truth, depths=(2,)).at_depth[0]
print(prediction)
print(ranking[:2], exact, group)
```

In this seeded example, test AUC is 0.686 and the top two proteins are B and A.
Recall, precision and group recovery are all 1.0. The example is deliberately small;
these values demonstrate the workflow rather than estimate performance in a real cohort.

The prediction result reports held-out ROC AUC. The recovery results report how many
of A and B were selected and how many of their groups were represented. For example,
selecting A and B_proxy recovers one of two contributors but both groups. AUC measures
prediction on people; recall and precision measure identification of proteins.

## Compare with matched chance

A random list has some chance of selecting a contributor. Here the measured pairs and
singletons form two strata. The reference keeps the selected count in each stratum,
then redistributes those selections among its members.

```python
nomination = ppm.nomination_from_scores(names, scores, depth=2)
strata = ("pair", "pair", "pair", "pair", "singleton", "singleton", "singleton", "singleton")
matched = ppm.evaluate_matched_recovery(nomination, train.truth, estimand="exact", strata=strata)
print(matched.observed_recovery, matched.expected_recovery, matched.chance_adjusted_recovery)
```

If both selections come from the four paired proteins, random assignment recovers half
the two contributors on average. Finding both gives recovery 1.0, expected recovery
0.5 and an excess of 0.5. Finding just A and B_proxy gives exact recovery 0.5 and zero
excess under this reference. Matching makes the comparison respect the types of proteins
selected rather than rewarding selection from an easier group.

## Change the biological hypothesis

The same measurements can test an outcome driven by individual proteins or by a shared
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
[outcome examples](outcomes.md) show how to extend this design to other clinical responses.
