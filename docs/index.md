# PyPlasmode

PyPlasmode lets you evaluate how well a modelling procedure predicts outcomes and identifies the
features behind them. You provide a numeric feature matrix, define a signal and simulate an
outcome. Because the generating features are known, you can compare a model's feature ranking
with them.

This combination of observed features and simulated outcomes is a *plasmode*. It preserves
relationships in your measurements while allowing you to vary the signal, effect size and
outcome frequency. The library supports biomarker benchmarking and accepts numeric tabular data
from other applications.

Use your existing tools for model fitting, tuning and explanation. PyPlasmode simulates the data
and evaluates the predictions, rankings and selected feature sets you return.

Start with the [feature-selection tutorial](tutorial.md) to fit a model, rank its features
and compare their recovery with random selection. The [visual gallery](gallery.md) shows
signal structures, recovery among correlated features and outcome families.

## Install and run

Install from [PyPI](https://pypi.org/project/pyplasmode/) with Python 3.12-3.15.

```bash
python -m pip install pyplasmode
```

For the survival and validation examples, add the optional estimators and plotting tools
with `python -m pip install "pyplasmode[validation]"`.

Three scripts run locally on synthetic inputs. The
[quickstart](https://github.com/BradSegal/PyPlasmode/blob/main/examples/quickstart.py) reports
test-set AUC (area under the ROC curve) and the fraction of generating features recovered. The
[survival example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/survival.py) sets
event risks at five and ten years and fits a Cox model to the generated follow-up. The
[custom example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/custom_signal.py)
uses a signal whose response plateaus at extreme feature values.

## A complete comparison

The **signal** is what you specify: which features generate the outcome, in what structure
and how strongly. The **generating score** is the number computed for each row from those
generating features. The **outcome** is the response generated from that score, with random
variation. **Recovery** is the fraction of the generating features in a selection. A
**panel** is a selected subset of features used to predict.

1. Supply a numeric matrix and stable feature identifiers.
2. Choose a [signal](truths.md) and an [outcome in observable units](outcomes.md).
3. Split source rows before resampling to keep training and test observations separate.
4. Generate training, validation and test samples, then fit and tune your models.
5. Evaluate predictions and [feature recovery](evaluation.md), including tied importance scores.
6. Repeat with independent seeds and summarise the paired results.

Compare methods on the same generated samples. Vary the signal, effect size, sample size, or
censoring to explore the conditions that matter for your application. The
[methods guide](methods.md) explains how these choices define the simulation.

## Return types

`Plasmode` contains `X`, `feature_ids`, `outcome`, `truth` (the signal and its generating
score) and `source_indices`.
`PartitionedPlasmode` contains training, validation and evaluation plasmodes and their original
source identities. Evaluation functions return dataclasses or, for a single metric, a number;
the [API reference](api.md) lists them.

For empirical inputs, follow the [data handling guidance](privacy-and-limitations.md).
