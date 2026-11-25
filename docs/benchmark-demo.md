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

The extra is tested on Python 3.12-3.15 on Linux, with Windows and macOS covered by CI on Python
3.13.

The default is 3,000 rows and 120 features, with one numerical thread. Allow a
few minutes on a laptop CPU. Use `--rows 1000` for a smaller run.
`--seed` sets the random seed (default 42). The final console table and optional
CSV contain all 27 comparisons: three signal structures, three target AUCs and three
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

The sparse and correlated signals use the same generating features. Group recovery
also credits selecting a correlate.
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

A fixed 60:40 split separates training and evaluation. Scaling is fitted on training features
only. Cox uses all observed event times with L2 penalty 1. Ridge uses ten-year event indicators
with penalty 10. XGBoost uses the same indicators with 80 depth-two trees, learning rate 0.05
and one CPU thread. No hyperparameter search is performed. Ridge and XGBoost labels are fully
observed only because there is no early censoring.

## Reading the output

`AUROC` measures how well a model ranks people with an event by ten years
above those observed beyond ten years. With no early censoring it equals
binary AUC on the ten-year event labels. With censoring, the
[AUC estimator](metrics.md#censored-discrimination) uses training follow-up
to give more weight to cases whose event times were less likely to be observed.

`Target` is the requested expected AUC of the generating score.
`Oracle` is the AUC of that score on the held-out rows after outcomes are drawn.
The CSV also gives the expected AUC that calibration achieved.

```text
Scenario     Target Method    AUROC Oracle Recovery Chance Adjusted Top20
sparse       0.60   Cox      0.529 0.606  0.125    0.167  -0.042    20
sparse       0.60   ridge    0.529 0.606  0.125    0.167  -0.042    20
sparse       0.60   XGBoost  0.526 0.606  0.250    0.167  0.083    20
```

In the Cox row, the panel recovers one of eight generating features (0.125).
Subtracting chance recovery (0.167 after rounding) gives `Adjusted` of −0.042.
Negative values mean recovery below chance. Compare `AUROC` and recovery
separately. Similar AUCs can accompany different recovery, as these rows show.
The results apply to one synthetic split with fixed model settings.

Cox and ridge rank features by absolute coefficients on the standardised
inputs. XGBoost ranks by gain importance. `Top20` counts features given any
selection weight. It falls below 20 when fewer than 20 features have positive
importance. It exceeds 20 when features share the remaining slots at
the cutoff. Zero-importance features get no weight.

`evaluate_matched_recovery` computes chance recovery by assigning selection
weights to features at random. It preserves the number of positive-importance
features and the pattern of full and fractional selection weights.
The example treats all features as one pool. See
[Matched chance recovery](evaluation.md#matched-chance-recovery).
