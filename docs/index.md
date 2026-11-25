# PyPlasmode

A model can predict an outcome well while ranking the wrong features first. On real data you
cannot tell, because nobody knows which features generated the outcome. PyPlasmode makes that
known: you provide a numeric feature matrix, choose which of its features generate an outcome,
and simulate the outcome. You can then measure how well a modelling procedure predicts the
outcome and how many of the generating features its ranking puts first.

Measured features combined with a simulated outcome are called a *plasmode*. Keeping the
measured features preserves their correlations, which is what makes selecting the right
features hard in practice, while the simulation lets you set the signal, its strength and how
common the outcome is. The library was built for biomarker studies and accepts any numeric
table of observations and named features.

PyPlasmode does not fit models. You fit, tune and explain them with your existing tools and
hand back their predictions, rankings or selected features for evaluation.

The [feature-selection tutorial](tutorial.md) is the place to start: it fits a model, ranks
its features and compares their recovery with random selection. The
[visual gallery](gallery.md) shows the signal structures, recovery among correlated features
and the outcome families as figures.

## Install and run

PyPlasmode needs Python 3.12 to 3.15. Install it from
[PyPI](https://pypi.org/project/pyplasmode/):

```bash
python -m pip install pyplasmode
```

The survival and validation examples fit models from other libraries. To run them, install the
optional estimators and plotting tools with `python -m pip install "pyplasmode[validation]"`.

Three scripts show a complete run on synthetic data, on your own machine:

- The [quickstart](https://github.com/BradSegal/PyPlasmode/blob/main/examples/quickstart.py)
  reports test-set AUC (area under the ROC curve) and the fraction of generating features
  recovered.
- The [survival example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/survival.py)
  sets event risks at five and ten years and fits a Cox model to the simulated follow-up.
- The [custom example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/custom_signal.py)
  uses a signal whose effect levels off at extreme feature values.

## A complete comparison

A benchmark compares what a procedure predicts and selects with the signal that generated the
outcome. It rests on five terms:

- The **signal** is what you specify: which features generate the outcome, in what structure
  and how strongly.
- The **generating score** is the number computed for each row from those generating features.
- The **outcome** is the response simulated from that score, with random variation.
- **Recovery** is the fraction of the generating features in a selection.
- A **panel** is a selected subset of features used to predict.

A study runs in six steps:

1. Supply a numeric matrix with a name for each feature. Recovery is scored by matching your
   model's feature names against the signal's, so the names must be the same in both.
2. Choose a [signal](truths.md) and an [outcome in observable units](outcomes.md), such as an
   event probability or a mean.
3. Split the source rows into training, validation and test pools before drawing samples.
   Otherwise the same measured row could appear in both the rows a model is fitted on and the
   rows it is judged on.
4. Generate a sample from each pool, then fit and tune your models on the training and
   validation samples.
5. Evaluate predictions on the test sample and [feature recovery](evaluation.md) from the
   rankings, including rankings with tied importance scores.
6. Repeat with independent seeds. One simulated dataset gives one draw of each result; repeats
   show how much it varies.

To compare methods, run each on the same generated samples, so that differences between them
come from the methods and not from the data. See [Methods](methods.md).

## Return types

`generate` simulates one dataset and returns a `Plasmode`. It holds the sampled measurements
(`X`, with column names in `feature_ids`), the simulated `outcome`, and in `truth` the
generating score and what a selection should recover.

`generate_partitioned` returns a `PartitionedPlasmode` holding three such datasets:
`training`, `validation` and `evaluation` (the test sample).

To link a simulated row back to your own data, for example to add a covariate, use its source
index: its zero-based row position in the matrix it was drawn from. In a `Plasmode` from
`generate`, `source_indices` points into the `Population` you supplied. In each dataset of a
`PartitionedPlasmode`, `source_indices` points into that dataset's pool. To point into the
original unsplit `Population` instead, use `training_source_indices`,
`validation_source_indices` or `evaluation_source_indices`.

The [API reference](api.md) lists every result field. Before using empirical measurements,
read the [data handling guidance](privacy-and-limitations.md), because simulated datasets can
reproduce individual records.
