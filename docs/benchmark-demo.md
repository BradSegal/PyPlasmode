# Prediction and feature recovery on synthetic data

[`examples/benchmark_recovery_demo.py`](https://github.com/BradSegal/PyPlasmode/blob/main/examples/benchmark_recovery_demo.py)
compares how well Cox, ridge and XGBoost predict an outcome and recover the features
that generated it. It generates all features and outcomes in memory and requires no
downloads.

On macOS, install the OpenMP runtime required by XGBoost with
`brew install libomp`. Then install the optional model dependencies and download
the linked example:

```bash
python -m pip install "pyplasmode[benchmark]"
python benchmark_recovery_demo.py --output benchmark-results.csv
```

The extra is tested on Python 3.12–3.15 on Linux, with Windows and macOS covered
by CI on Python 3.13.

The default is 3,000 rows and 120 features, with one numerical thread. Allow a
few minutes on a laptop CPU. `--rows 1000` provides a smaller demonstration;
`--seed` sets the random seed (default 42). The final console table and optional
CSV contain all 27 comparisons: three signal structures, three effect sizes and three
methods. The CSV retains full precision, the expected and realised AUC (area under
the ROC curve) of the generating score,
the recovery definition and the achieved selection size.

## What is compared

Features are Gaussian, arranged in 30 independent blocks of four. Features in
the same block have population correlation 0.7. There are three signal structures:

| Signal structure | Generating features | Recovery at depth 20 |
| --- | --- | --- |
| Sparse | Eight generating features, one in each of eight blocks | Fraction of generating features selected |
| Correlated | The same eight features (`sentinels`), each with three correlates: the other features in its block | Fraction of groups represented by at least one selected member |
| Distributed | Equal weights on the first 60 features | Fraction of absolute generating weight selected |

The sparse and correlated signals use the same generating features; their different
recovery definitions show how crediting correlates changes the question being answered.
A 20-feature panel cannot cover all 60 distributed features; its maximum recovery,
before subtracting chance, is one third. Interpret recovery within a signal structure,
not between structures.

`TimeToEventOutcome(target_auc=...)` sets the expected ten-year AUC of the generating
score, the number computed for each row from the generating features, to
0.60, 0.70 and 0.80 in the synthetic population. Ten-year event
risk is 0.20 and eleven-year risk is 0.22. Follow-up ends at eleven years,
with no earlier censoring. Calibration precedes outcome sampling, so the realised
held-out value varies around the request. Common random draws
are used across settings.

A fixed 60:40 split separates training and evaluation. Scaling is learnt on
training features only. Cox uses all observed event times with L2 penalty 1.
Ridge uses ten-year event indicators with penalty 10. XGBoost uses the same
indicators, 80 depth-two trees, learning rate 0.05 and one CPU thread. There is
no hyperparameter search. The ridge and XGBoost labels are fully observed only
because there is no early censoring.

## Reading the output

`AUROC` is the ten-year cumulative/dynamic AUC from
`pyplasmode.metrics.cumulative_dynamic_auc`, using the training censoring
distribution. With the chosen follow-up this
equals ordinary binary AUC at ten years.

Absolute standardised coefficients rank Cox and ridge features; XGBoost uses
gain importance. `nomination_from_scores` allocates boundary ties fractionally
and does not fill unused slots with zero-importance features. `Top20` reports
the achieved size. `evaluate_matched_recovery` compares each selection with
random selections of the same size, positive support and tie
geometry, drawn without reference to the signal, within one common feature stratum:
all blocks have the same size and population correlation.

`Adjusted` is recovery above chance: **observed minus expected chance recovery**,
without rescaling
or clipping. Negative values are possible. `Target` is the requested expected AUC of the
generating score; the CSV also records the value that calibration achieved.
`Oracle` is its realised held-out value. Compare `AUROC` and recovery
across the three targets: stronger discrimination need not yield an equally
large change in recovery. This is one synthetic split with fixed model settings,
so its table demonstrates the evaluation rather than establishing a method ranking.
