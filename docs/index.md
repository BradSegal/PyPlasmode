# PyPlasmode

PyPlasmode benchmarks the reliability of biomarker discovery. It generates outcomes from
specified effects within your feature measurements, making the relevant contributors known.
You can then measure how accurately a model predicts and how reliably its explanation
identifies those contributors. This turns a plausible protein ranking into a testable
selection procedure.

Explore the [visual gallery](gallery.md) for examples of signal shapes, correlated-feature
recovery and outcome families.

Start with the [biomarker tutorial](tutorial.md) to fit a model, rank its proteins and
compare recovery with a reference matched for correlated groups.

## Install and run

Install from [PyPI](https://pypi.org/project/pyplasmode/) with Python 3.12-3.14.
The validation extra includes the optional dependencies used by the examples:

```bash
python -m pip install "pyplasmode[validation]"
```

Start with the [quickstart](https://github.com/BradSegal/PyPlasmode/blob/main/examples/quickstart.py),
which reports test-set AUC and the fraction of generating features recovered.
The [survival example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/survival.py)
sets event risks at five and ten years and fits a Cox model to the generated follow-up.
The [custom example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/custom_signal.py)
uses a signal whose response levels off at extreme feature values. This lets you test a
nonlinear relationship without changing the model-evaluation workflow. Download and run
these scripts locally; all three supply synthetic inputs.

## A complete comparison

1. Supply a numeric matrix and stable feature identifiers.
2. Choose a [generating signal](truths.md) and an [outcome in observable units](outcomes.md).
3. Partition source participants before resampling if prediction will be evaluated held out.
4. Generate training, validation and test samples, then fit and tune your models.
5. Pass predictions, complete rankings or fractional nominations to the [evaluators](evaluation.md).
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
