# PyPlasmode

A model can predict an outcome well while ranking the wrong features first. On real data, you
usually cannot tell because the features that generated the outcome are unknown. With
PyPlasmode, you provide a numeric feature matrix, choose which of its features generate an
outcome and simulate the outcome. You can then measure how well a modelling procedure predicts
the outcome and how many of the generating features it ranks first.

A dataset of measured features combined with a simulated outcome is called a *plasmode*. Drawing
whole rows preserves how measured features vary together, which is what makes it hard to select
the right features in practice. The simulation allows you to set the signal, its strength and
how common the outcome is. The library was built for biomarker studies. It accepts a non-empty
numeric table of observations and uniquely named features. Use NaN for missing values. Infinite
values are rejected.

PyPlasmode does not fit models. You fit them with your own tools and pass back their
predictions, rankings or selected features.

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
outcome.

- The **signal** is what you specify: which features generate the outcome, in what structure
  and how strongly.
- The **generating score** is the number computed for each row from those generating features.
- The **outcome** is the response simulated from that score, with random variation.
- **Recovery** is the fraction of the generating features in a selection.
- A **panel** is a selected subset of features used to predict.

1. Supply a numeric matrix with a name for each feature. Recovery is scored by matching your
   model's feature names against the signal's, so the names must be the same in both.
2. Choose a [signal](truths.md) and an [outcome in observable units](outcomes.md), such as an
   event probability or a mean.
3. Split the source rows into training, validation and test pools before drawing samples.
   Otherwise the same measured row could appear in both the rows a model is fitted on and the
   rows it is judged on.
4. Generate a sample from each pool. Fit on the training sample and tune on the validation
   sample. Reserve the test sample for evaluation.
5. Evaluate predictions on the test sample and [feature recovery](evaluation.md) from the
   rankings. Where importance scores tie, score the tied features together, so that an
   arbitrary order among them does not change recovery.
6. Repeat with independent seeds to measure how much each result varies between simulated
   datasets.

To compare methods, run each on the same generated samples so that differences between them come
from the methods and not from the data. See [Methods](methods.md).

## Return types

A generated dataset gives you the outcomes to fit, the targets to score against and the
indices that link each row to your source data.

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

Before using empirical measurements, read the
[data handling guidance](privacy-and-limitations.md) because simulated datasets can reproduce
individual records.
