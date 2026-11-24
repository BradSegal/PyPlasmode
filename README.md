# PyPlasmode

PyPlasmode tests prediction and feature selection against a known signal. It combines your
feature data with simulated outcomes to measure how accurately a model predicts them and how
well it identifies the generating features used to produce them.

A plasmode uses observed measurements to maintain realistic feature relationships while you
control how the outcome is generated. The library was built for biomarker benchmarking and
accepts any numeric matrix of observations and named features, including the synthetic matrices
below.

You can integrate it with your existing modelling workflow. Generate a dataset, fit and tune
your models, then evaluate their predictions, feature rankings, or selected panels.

[![From observed measurements to simulated outcomes, model predictions and feature recovery](https://raw.githubusercontent.com/BradSegal/PyPlasmode/main/docs/assets/concepts/plasmode-workflow.png)](https://bradsegal.github.io/PyPlasmode/methods/)

The [visual gallery](https://bradsegal.github.io/PyPlasmode/gallery/) shows signal structures, recovery among correlated
features and the five outcome families. The [feature-selection tutorial](https://bradsegal.github.io/PyPlasmode/tutorial/)
follows a complete example from correlated measurements through model fitting, attribution
and recovery above chance.

## Installation

Requires Python 3.12-3.15. Install from [PyPI](https://pypi.org/project/pyplasmode/):

```bash
python -m pip install pyplasmode
```

The runtime dependencies are NumPy, SciPy and scikit-learn. To include the
reference validation and survival examples:

```bash
python -m pip install "pyplasmode[validation]"
```

## Ten-minute example

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
import pyplasmode as ppm

rng = np.random.default_rng(1)
population = ppm.Population(
    rng.normal(size=(2_000, 100)),
    tuple(f"feature_{index}" for index in range(100)),
)

partition = ppm.partition_population(
    population,
    training_fraction=0.6,
    validation_fraction=0.2,
    seed=7,
)
samples = ppm.generate_partitioned(
    partition,
    truth=ppm.SparseTruth(features=("feature_3", "feature_17", "feature_42")),
    outcome=ppm.BinaryOutcome(probability=0.15, odds_ratio=2.0),
    sample_sizes=ppm.PartitionSampleSizes(1_200, 400, 400),
    sampling_method="without_replacement",
    seed=42,
)

train, test = samples.training, samples.evaluation
model = LogisticRegression(max_iter=1_000).fit(train.X, train.outcome.values)
ranking = tuple(train.feature_ids[index] for index in np.argsort(-np.abs(model.coef_[0])))

recovery = ppm.evaluate_ranking(ranking, train.truth, depths=(10, 20, 50))
prediction = ppm.evaluate_prediction(
    test.outcome.values,
    model.predict_proba(test.X)[:, 1],
    metric="roc_auc",
)
print(recovery.at_depth[0].recall)
print(prediction.value)
```

The first number is the recovery (`recall` in the API): the fraction of the three generating
features in the model's top ten. The second is the AUC (area under the ROC curve) on held-out
observations. Repeat the comparison with different seeds to estimate average performance and
its Monte Carlo uncertainty. `compare_paired_replicates` matches two procedures' results by
simulated dataset before computing paired differences, and counts failures among the matched
pairs. See
[paired evaluation](https://bradsegal.github.io/PyPlasmode/evaluation/#paired-procedure-comparisons).

Source rows are split before resampling to keep training, validation and test rows disjoint.
This example uses a fixed model. The validation sample is available for tuning hyperparameters.

## What can be varied

- **Signal:** a null, sparse, correlated, distributed or interaction structure, or a custom
  function.
- **Outcome:** binary probability, continuous mean and spread, count rate, ordinal category
  probabilities or cumulative event risks at chosen times. Binary and time-to-event outcomes
  can take a target AUC of the generating score in place of an odds or hazard ratio
  ([calibration](https://bradsegal.github.io/PyPlasmode/calibration/)).
- **Evaluation:** feature and group recovery, weighted-module and graph-region coverage,
  reconstruction of the generating score, prediction accuracy, panel performance and ranking
  stability. Further functions handle tied scores and incomplete selections and compute
  recovery above chance.
  [`pyplasmode.metrics`](https://bradsegal.github.io/PyPlasmode/metrics/) provides selection
  stability and consensus across refits, AUC under censoring, paired bootstrap differences and
  the gap between a fitted model's AUC and that of the generating score.

Start with [Methods](https://bradsegal.github.io/PyPlasmode/methods/), then choose a [signal](https://bradsegal.github.io/PyPlasmode/truths/) and
[outcome](https://bradsegal.github.io/PyPlasmode/outcomes/). The [evaluation guide](https://bradsegal.github.io/PyPlasmode/evaluation/) defines each result.
The [validation example](https://bradsegal.github.io/PyPlasmode/validation/) checks the generators and recovery metrics against
known values and reference estimators. The [documentation](https://bradsegal.github.io/PyPlasmode/)
also holds the API reference.

The executable [quickstart](https://github.com/BradSegal/PyPlasmode/blob/main/examples/quickstart.py),
[survival example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/survival.py),
[custom signal](https://github.com/BradSegal/PyPlasmode/blob/main/examples/custom_signal.py)
and [metrics and calibration example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/metrics_calibration.py)
use synthetic inputs. The [synthetic benchmark](https://bradsegal.github.io/PyPlasmode/benchmark-demo/)
fits Cox, ridge and XGBoost across three signal structures and three target AUCs, then
compares held-out AUC with recovery above chance in the top 20 features. It needs the
optional model libraries: `python -m pip install "pyplasmode[benchmark]"`.

## Using it with a coding agent

The folder [`skills/pyplasmode`](https://github.com/BradSegal/PyPlasmode/tree/main/skills/pyplasmode)
is a skill package that teaches a coding agent the workflow, the choices and the common
mistakes. See [Coding agents](https://bradsegal.github.io/PyPlasmode/agents/).

## Working with empirical data

Resampling preserves the source data's feature relationships and missingness patterns.
It can reproduce individual records, so generated datasets require the same privacy protections
as their source. See [Data handling](https://bradsegal.github.io/PyPlasmode/privacy-and-limitations/).

## Licence

PyPlasmode uses the [BSD 3-Clause License](https://github.com/BradSegal/PyPlasmode/blob/main/LICENSE).
