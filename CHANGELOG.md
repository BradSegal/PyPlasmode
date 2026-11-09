# Changelog

## 1.2.0 — 2026-11-09

### Added

- `pyplasmode.metrics`: chance-corrected selection stability with unequal budgets
  and ties, tie-aware RBO, inclusion-first consensus ranks, IPCW cumulative/dynamic
  AUROC, paired participant bootstrap differences and oracle-minus-fitted
  `discrimination_gap` (also available as `learnability_gap`).
- `target_auc` for binary and survival outcomes, reporting expected oracle AUROC
  and the solved odds/hazard ratio while retaining marginal risk.
- Metrics and calibration guides, executable examples and a synthetic Cox,
  ridge and XGBoost benchmark comparing held-out AUROC and chance-adjusted recovery.
  Model dependencies are supplied by the optional `benchmark` extra.
- Identity-aligned paired replicate comparisons with explicit failed/unmatched
  pair counts and conditional Monte Carlo uncertainty.
- `allow_subset=True` for fractional recovery, matched recovery and outcome
  specificity, preserving generating denominators and the available-panel reference.
- Python 3.15 support and NumPy string arrays as bootstrap participant IDs.

### Changed

- RBO defaults to Webber whole-tie prefixes; independent tie resolution remains
  available through `tie_method="independent"`.
- Censored AUROC defaults to exact score ties. Use `tied_tolerance=1e-8` to
  reproduce scikit-survival's default score-tie convention.
- Default survival AUROC horizons precede the administrative cutoff; an explicit
  later administrative cutoff is supported.
- Calibration parameters support direct lookup, and `metrics` is discoverable
  through the root package. Unattainable target AUROCs raise explicitly.
- The validation extra requires statsmodels 0.14.6 or later. Source distributions
  omit the unreferenced concept PDFs, which remain in the repository.

### Fixed

- Never-selected consensus features sort below selected features even with wide ties.
- AUROC-targeted calibration tolerates single extreme signal observations and
  rejects requests above the supported effect bracket without clipping.
- Edgeless graph regions report unavailable edge coverage and expose the target
  edge count while retaining node recovery.
- Panel restrictions retain module weights, group proxies and joint interaction
  requirements rather than redefining the generating truth.

## 1.0.0

First public release. PyPlasmode provides outcome generation, feature-recovery evaluation,
worked examples and reference validation for biomarker benchmarking.

### Included in this release

- Empirical complete-row resampling and source-partitioned generation.
- Null, sparse, correlated-group, module, interaction and custom signals.
- Five outcome families calibrated in observable units.
- Recovery, prediction, panel, reconstruction and rank-stability evaluation.
- API documentation, worked examples, validation figures and cross-platform CI.
- A visual gallery of signal shapes, correlated-feature recovery and outcome families,
  with editable vector figures and an executable source-data example.
- Python 3.12-3.14 and scikit-learn 1.6 through the current 1.x series.
- BSD 3-Clause licensing.
