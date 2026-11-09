# Prediction and biomarker recovery on synthetic data

[`examples/benchmark_recovery_demo.py`](https://github.com/BradSegal/PyPlasmode/blob/main/examples/benchmark_recovery_demo.py)
is a small, self-contained comparison of Cox, ridge and XGBoost. It generates
all features and outcomes in memory; it requires no participant data or downloads.

Install the optional model dependencies and download the linked example:

```bash
python -m pip install "pyplasmode[benchmark]"
python benchmark_recovery_demo.py --output benchmark-results.csv
```

The core package does not require scikit-survival or XGBoost. The extra is
tested on Python 3.12–3.15 on Linux, with Windows and macOS covered by CI on
Python 3.13; installation depends on the model libraries' platform support.

The default is 3,000 rows and 120 features, with one numerical thread. Allow a
few minutes on a laptop CPU. `--rows 1000` provides a smaller demonstration;
`--seed` sets the random seed (default 42). The final console table and optional
CSV contain all 27 comparisons: three mechanisms, three effect sizes and three
methods. The CSV retains full precision, expected and realised oracle AUROC,
the recovery definition and the achieved nomination size.

## What is compared

Features are Gaussian, arranged in 30 independent blocks of four. Features in
the same block have population correlation 0.7. There are three signals:

| Signal | Generating mechanism | Recovery at depth 20 |
| --- | --- | --- |
| Sparse | Eight direct features, one in each of eight blocks | Fraction of direct features selected |
| Correlated | Eight direct sentinels, each with three admissible correlated substitutes | Fraction of groups represented by at least one selected member |
| Distributed | Equal weights on the first 60 features | Fraction of absolute generating weight selected |

The sparse and correlated signals use the same generating features. Their
different recovery definitions show how allowing correlated substitutes changes
the question being answered. A 20-feature panel cannot cover all 60 distributed
features; its maximum unadjusted weight coverage is one third. Recovery columns
therefore measure each stated mechanism and should be interpreted within it.

`TimeToEventOutcome(target_auc=...)` sets expected ten-year oracle AUROC to
0.60, 0.70 and 0.80 in the finite synthetic covariate population. Ten-year event
risk is 0.20 and eleven-year risk is 0.22. Follow-up ends at eleven years,
with no earlier censoring. Calibration precedes outcome sampling, so realised
held-out oracle AUROC varies around the requested value. Common random draws
are used across settings.

A fixed 60:40 split separates training and evaluation. Scaling is learnt on
training features only. Cox uses all observed event times with L2 penalty 1.
Ridge uses ten-year event indicators with penalty 10. XGBoost uses the same
indicators, 80 depth-two trees, learning rate 0.05 and one CPU thread. There is
no hyperparameter search. For ridge and XGBoost the horizon labels are fully
observed because there is no early censoring; this construction would need to
change if early censoring were introduced.

## Reading the output

AUROC is the ten-year cumulative/dynamic estimate from
`pyplasmode.metrics.cumulative_dynamic_auc`, using the training censoring
distribution. Higher scores mean higher risk. With the chosen follow-up this
equals ordinary binary AUROC at ten years.

Absolute standardised coefficients rank Cox and ridge features; XGBoost uses
gain importance. `nomination_from_scores` allocates boundary ties fractionally
and does not fill unused slots with zero-importance features. `Top20` reports
the achieved size. `evaluate_matched_recovery` compares each nomination to
truth-blind random nominations with the same size, positive support and tie
geometry, within one common feature stratum. All blocks have the same size and
population correlation in this example.

`Adjusted` is **observed minus expected chance recovery**, without rescaling
or clipping. Negative values are possible. `Target` is the expected oracle AUROC requested and achieved by calibration.
The `Oracle` column is the realised
held-out AUROC of the generating signal. Compare fitted AUROC and recovery
across the three targets: stronger discrimination need not yield an equally
large change in recovery. This is one synthetic split with fixed model settings,
so its table demonstrates the evaluation rather than establishing a method ranking.
