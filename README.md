# PyPlasmode

[![PyPI](https://img.shields.io/pypi/v/pyplasmode)](https://pypi.org/project/pyplasmode/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyplasmode)](https://pypi.org/project/pyplasmode/)
[![Tests](https://github.com/BradSegal/PyPlasmode/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BradSegal/PyPlasmode/actions/workflows/ci.yml)
[![Documentation](https://github.com/BradSegal/PyPlasmode/actions/workflows/docs.yml/badge.svg?branch=main)](https://bradsegal.github.io/PyPlasmode/)
[![Licence](https://img.shields.io/pypi/l/pyplasmode)](https://github.com/BradSegal/PyPlasmode/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24518248.svg)](https://doi.org/10.5281/zenodo.24518248)

A model can predict an outcome well while ranking the wrong features first. In real data, you
usually cannot tell because you do not know which features generated the outcome. PyPlasmode
keeps your feature data and simulates the outcome from features you choose. You can then measure
both how well a model predicts the outcome and how many of the generating features it finds.

A dataset of measured features with a simulated outcome is called a *plasmode*. Sampling whole
rows keeps the relationships between measured features, which is what makes finding the right
ones hard. The library was made for biomarker benchmarking. It takes a non-empty numeric matrix
with one observation per row and a unique name for each feature. Use NaN for missing values.
Infinite values are rejected. It does not fit models. You fit them with your own tools and give
back their predictions, rankings or selected panels.

[![From observed measurements to simulated outcomes, model predictions and feature recovery](https://raw.githubusercontent.com/BradSegal/PyPlasmode/main/docs/assets/concepts/plasmode-workflow.png)](https://bradsegal.github.io/PyPlasmode/methods/)

Start with the [feature-selection tutorial](https://bradsegal.github.io/PyPlasmode/tutorial/)
or the [visual gallery](https://bradsegal.github.io/PyPlasmode/gallery/).

## Installation

Requires Python 3.12-3.15. Install from [PyPI](https://pypi.org/project/pyplasmode/):

```bash
python -m pip install pyplasmode
```

The runtime dependencies are NumPy, SciPy and scikit-learn. The validation and survival
examples fit models from other libraries; to run them, install the optional extra:

```bash
python -m pip install "pyplasmode[validation]"
```

## Ten-minute example

Fit a classifier on simulated training outcomes, then check whether it predicts held-out
outcomes and ranks the known generating features first.

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
print(round(prediction.value, 3))
# 1.0
# 0.626
```

The rows are split 60:20:20 into training, validation and test pools before sampling, so no row a
model is fitted on is also used to judge it. Three of the 100 features generate a binary outcome
that 15% of rows have on average, with odds that double for each standard deviation of the
generating score (the number computed for each row from the generating features).

The model ranks features by the size of its coefficients. The first number is the recovery
(`recall` in the API): the fraction of the three generating features in the model's top ten.
The second is the AUC (area under the ROC curve) on the test sample. The example uses fixed
settings. If you tune them, use the validation sample and reserve the test sample for scoring
the tuned model. Repeat the simulation with different seeds to estimate average performance
and its uncertainty.
[Paired evaluation](https://bradsegal.github.io/PyPlasmode/evaluation/#paired-procedure-comparisons)
compares two procedures across such repeats.

## What can be varied

Vary the signal, the outcome and the evaluation to match the conditions your method will face.

- **Signal:** a null, sparse, correlated, distributed or interaction structure, or a custom
  function.
- **Outcome:** binary probability, continuous mean and spread, count rate, ordinal category
  probabilities or cumulative event risks at chosen times. For binary and time-to-event outcomes,
  set a target AUC of the generating score in place of an odds or hazard ratio
  ([calibration](https://bradsegal.github.io/PyPlasmode/calibration/)).
- **Evaluation:** prediction accuracy, feature and group recovery, reconstruction of the
  generating score and stability across refits. Recovery can be compared with what random
  selection would achieve (recovery above chance), including when importance scores tie.

[Methods](https://bradsegal.github.io/PyPlasmode/methods/) describes how datasets are generated.
See also the [signal](https://bradsegal.github.io/PyPlasmode/truths/),
[outcome](https://bradsegal.github.io/PyPlasmode/outcomes/) and
[evaluation](https://bradsegal.github.io/PyPlasmode/evaluation/) guides, the
[validation example](https://bradsegal.github.io/PyPlasmode/validation/) and the
[API reference](https://bradsegal.github.io/PyPlasmode/api/). The
[synthetic benchmark](https://bradsegal.github.io/PyPlasmode/benchmark-demo/) compares held-out
AUC with top-20 recovery above chance for Cox, ridge and XGBoost.

## Using it with a coding agent

Use [`skills/pyplasmode`](https://github.com/BradSegal/PyPlasmode/tree/main/skills/pyplasmode) to
give a coding agent the simulation workflow, design choices and common pitfalls. See [Coding
agents](https://bradsegal.github.io/PyPlasmode/agents/).

## Working with empirical data

Resampling draws whole rows, so each generated row keeps its source row's measurements and missing
values, and correlations and missingness rates vary only by sampling. It can reproduce individual
records, so generated datasets require the same privacy protections as their source. See [Data
handling](https://bradsegal.github.io/PyPlasmode/privacy-and-limitations/).

## Citing

If you use PyPlasmode, please cite the version you used. For version 1.2.1:

> Segal B, Fieggen J, Simond G, Belgrave D, Clifton D, Clifton L. PyPlasmode (version 1.2.1).
> Zenodo; 2026. <https://doi.org/10.5281/zenodo.24518249>

```bibtex
@software{segal2026pyplasmode,
  author    = {Segal, Bradley and Fieggen, Joshua and Simond, Greg and Belgrave, Danielle and
               Clifton, David and Clifton, Lei},
  title     = {PyPlasmode},
  version   = {1.2.1},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24518249},
  url       = {https://doi.org/10.5281/zenodo.24518249}
}
```

Each release is archived on Zenodo with its own DOI, and
[10.5281/zenodo.24518248](https://doi.org/10.5281/zenodo.24518248) always resolves to the latest.
On GitHub, "Cite this repository" in the sidebar gives APA and BibTeX forms from
[`CITATION.cff`](https://github.com/BradSegal/PyPlasmode/blob/main/CITATION.cff), with the DOI
for all versions.

## Licence

PyPlasmode uses the [BSD 3-Clause
License](https://github.com/BradSegal/PyPlasmode/blob/main/LICENSE).
