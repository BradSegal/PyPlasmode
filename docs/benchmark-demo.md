# Prediction and feature recovery on synthetic data

A model that predicts well is often taken to have found the features that drive the outcome.
[`examples/benchmark_recovery_demo.py`](https://github.com/BradSegal/PyPlasmode/blob/main/examples/benchmark_recovery_demo.py)
tests that assumption where the answer is known. Cox, ridge and XGBoost each predict a time-to-event
outcome and select 20 features, and the script compares their prediction with how many generating
features they select. It generates all features and outcomes in memory, so it needs no downloads.

XGBoost needs the OpenMP runtime, which macOS lacks; on a Mac, install it first with
`brew install libomp`. Then install the optional model libraries, download the linked script
and run it:

```bash
python -m pip install "pyplasmode[benchmark]"
python benchmark_recovery_demo.py --output benchmark-results.csv
```

The optional libraries are tested on Python 3.12 to 3.15 on Linux, and on Python 3.13 on
Windows and macOS.

The default run uses 3,000 rows on one processor thread and takes a few minutes on a laptop.
`--rows 1000` gives a faster, smaller run, and `--seed` changes the random seed from its default
of 42. The optional CSV holds the results at full precision.

## What is compared

The 120 features are Gaussian, arranged in 30 independent blocks of four. Features in the same
block have a population correlation of 0.7, so each feature has three correlates that a model
could select in its place.

Each signal structure decides which features generate the outcome and what a 20-feature
selection has to contain to recover them:

| Signal structure | Generating features | Recovery at depth 20 |
| --- | --- | --- |
| Sparse | Eight generating features, one in each of eight blocks | Fraction of generating features selected |
| Correlated | The same eight features (`sentinels`), each with three correlates: the other features in its block | Fraction of groups represented by at least one selected member |
| Distributed | Equal weights on the first 60 features | Fraction of absolute generating weight selected |

The sparse and correlated structures produce the same generating score from the same eight
features, so their outcomes and fitted models are identical. Only what counts as recovery
differs: group recovery also credits a correlate. In the distributed structure, 20 selected
features can carry at most 20 of the 60 equal weights, so its recovery cannot exceed a third
before chance is subtracted. Since each structure has its own measure and maximum, compare
recovery within a structure, not between structures.

Ten-year event risk is 0.20 and eleven-year risk is 0.22. Follow-up ends at eleven years, and nobody
leaves earlier. Prediction difficulty is set through `TimeToEventOutcome(target_auc=...)`: the
expected ten-year AUC of the generating score (the number computed for each row from the generating
features) is 0.60, 0.70 or 0.80. The outcome is simulated after this calibration, so the AUC
realised on the test rows varies around the requested value. All nine settings use the same
feature rows, split and underlying random numbers, so the comparisons hold those inputs
fixed. Running the benchmark with another seed can change the differences.

A fixed random split puts 60% of rows in training and 40% in evaluation. Features are scaled
using the training rows only, so the evaluation rows do not inform the fit. The three methods:

- **Cox** fits a proportional hazards model to the observed event times, with L2 penalty 1.
- **Ridge** fits a penalised linear regression (penalty 10) to a yes or no label for an event
  by ten years.
- **XGBoost** fits 80 trees of depth two to the same label, with learning rate 0.05 and one
  processor thread.

The ten-year label is known for every row only because no one leaves follow-up before ten years.
With censoring before ten years, ridge and XGBoost would need another approach. There is no
hyperparameter tuning, so the results are for these settings and do not show each method at its
best.

## Reading the output

The console table has one row per comparison:

- `AUROC` is the model's ten-year AUC on the evaluation rows: how well it ranks people with an
  event by ten years above those still event-free at ten years. With no early censoring, this
  equals ordinary AUC on the ten-year labels. With censoring, the
  [AUC estimator](metrics.md#censored-discrimination) would use the training follow-up to give
  more weight to cases whose event times were less likely to be observed.
- `Target` is the requested expected AUC of the generating score, and `Oracle` is that score's
  AUC on the evaluation rows once the outcomes are drawn. `Oracle` is roughly the best `AUROC`
  a model could reach on these rows; a fitted model can exceed it slightly by chance.
- `Recovery` uses the recovery measure for the signal structure on the top 20 features.
  `Chance` is the expected recovery when a selection of the same size (normally 20 features)
  is assigned to features at random. `Adjusted` is recovery above chance, the difference
  between the two.
- `Top20` is the number of features counted as selected.

The first three rows of the default run:

```text
Scenario     Target Method    AUROC Oracle Recovery Chance Adjusted Top20
sparse       0.60   Cox      0.529 0.606  0.125    0.167  -0.042    20
sparse       0.60   ridge    0.529 0.606  0.125    0.167  -0.042    20
sparse       0.60   XGBoost  0.526 0.606  0.250    0.167  0.083    20
```

In the Cox row, the top 20 contain one of the eight generating features, a recovery of 0.125.
Twenty features chosen at random from 120 would contain 20/120, or 0.167, of them on average,
so recovery above chance is −0.042: Cox found fewer generating features than random selection
would. XGBoost predicts no better than Cox (AUROC 0.526 against 0.529) but selects two
generating features, a recovery above chance of 0.083. Similar AUC can come with different
recovery, so read the two columns separately. All rows come from one split with fixed model
settings and carry no uncertainty intervals.

The CSV adds the expected AUC the calibration achieved and the recovery measure used.

Cox and ridge rank features by the absolute size of their coefficients on the scaled inputs,
and XGBoost by gain importance. Features with zero importance are never selected, so `Top20`
falls below 20 when fewer than 20 features have positive importance. When several features
tie for the last places, they share them, each counted as a fraction of a selection, so
`Top20` can exceed 20.

`Chance` comes from `evaluate_matched_recovery`, which assigns the selection probabilities to
features at random while keeping the number of selected features and their fractional shares.
The script uses one stratum for all 120 features because each belongs to a block of four with
the same within-block correlation, so no feature has an advantage due to its correlation
structure. See
[Matched chance recovery](evaluation.md#matched-chance-recovery).
