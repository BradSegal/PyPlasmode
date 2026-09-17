# PyPlasmode

PyPlasmode lets you test how well a modelling procedure predicts outcomes and identifies
the features behind them. You supply a numeric feature matrix, specify a signal and generate
an outcome. Because the generating features are known, you can compare a model's feature
ranking with the mechanism that produced the data.

This combination of observed features and simulated outcomes is a *plasmode*. It retains
the relationships in your measurements while letting you vary the signal, effect size and
outcome frequency. The library was developed for biomarker benchmarking, but accepts numeric
tabular data from other applications. The examples use synthetic matrices so they can run
without an external dataset.

Keep model fitting, tuning and explanation in your usual tools. PyPlasmode generates the
data and evaluates the predictions, rankings and selected feature sets you return.

Explore the [visual gallery](gallery.md) for examples of signal shapes, correlated-feature
recovery and outcome families.

Start with the [feature-selection tutorial](tutorial.md) to fit a model, rank its features
and compare their recovery with random selection.

## Install and run

Install from [PyPI](https://pypi.org/project/pyplasmode/) with Python 3.12-3.14.

```bash
python -m pip install pyplasmode
```

For the survival and validation examples, add the optional estimators and plotting tools
with `python -m pip install "pyplasmode[validation]"`.

Start with the [quickstart](https://github.com/BradSegal/PyPlasmode/blob/main/examples/quickstart.py),
which reports test-set AUC and the fraction of generating features recovered.
The [survival example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/survival.py)
sets event risks at five and ten years and fits a Cox model to the generated follow-up.
The [custom example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/custom_signal.py)
uses a signal whose response levels off at extreme feature values. This lets you test a
nonlinear relationship without changing the model-evaluation workflow. Download and run
these scripts locally; all three supply synthetic inputs.

## A complete comparison

The **signal** is the score calculated from your chosen features. The **outcome** is the
response generated from that score, with random variation. **Recovery** measures how well
the selected features match the specified signal. A **panel** is simply a selected subset
of features, whether or not they are biomarkers.

1. Supply a numeric matrix and stable feature identifiers.
2. Choose a [generating signal](truths.md) and an [outcome in observable units](outcomes.md).
3. Split source rows before resampling to keep training and test observations separate.
4. Generate training, validation and test samples, then fit and tune your models.
5. Evaluate predictions and [feature recovery](evaluation.md), including tied importance scores.
6. Repeat with independent seeds and summarise the paired results.

Compare methods on the same generated samples. Vary the signal, effect size, sample size
or censoring to explore the conditions that matter for your application. The
[methods guide](methods.md) explains how these choices define the simulation.

## Return types

`Plasmode` contains `X`, `feature_ids`, `outcome`, `truth` and `source_indices`.
`PartitionedPlasmode` contains training, validation and evaluation plasmodes and their original
source identities. Evaluation functions return dataclasses; `dataclasses.asdict(result)`
converts them to dictionaries for analysis or export.

For empirical inputs, follow the [data handling guidance](privacy-and-limitations.md).
