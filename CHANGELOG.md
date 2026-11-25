# Changelog

## Unreleased

Documentation only. No executable code changed.

### Changed

- Every guide section says what it is for, gives the reason for each instruction and walks
  through its examples with their printed output. The tutorial reads as one lesson.
- Public docstrings say what each name is and what each argument and result field means.
- The references page lists plain citations.
- The contributing guide separates everyday checks from release checks.
- The examples of two more pages (signals and methods) run in the test suite.

### Fixed

- The front page said a single-metric evaluator returns a number; it returns a result object
  with the number in `value`.
- The prose for `evaluate_panel` named the panel and the reference in the opposite order to the
  function's arguments.
- `failure_count` in `summarize_replicates` was described as keeping failed runs in the summary;
  it records the count, and the mean and interval use successful runs only.
- The benchmark page described its three target AUCs as effect sizes.
- The calibration page said a fitted model's held-out AUC is typically below that of the
  generating score; in a finite sample the difference can take either sign.
- The API page said latent survival times beyond administrative follow-up are infinite; they are
  infinite when the event does not occur by the last risk point.
- Graph edges were described as weighted in evaluation; every edge counts equally.
- Generation was described as sampling complete rows; it draws whole rows, missing values
  included.
- The realisation tolerances for ordered categories and for survival event and censoring risks
  are stated.

## 1.2.1 — 2026-11-25

A documentation release. The library computes the same results as 1.2.0: no executable code
in the package changed apart from the version string.

### Added

- A skill package for coding agents (`skills/pyplasmode`): the workflow, the choice of signal
  structure and recovery measure, common mistakes and five tested recipes.

### Changed

- The documentation uses one set of terms throughout: generating features, generating score,
  signal structure, correlates, recovery above chance and AUC. Names in code are unchanged.
- Docstrings and comments use the same terms.
- `CITATION.cff` carries an abstract for archive metadata.

### Fixed

- Descriptions that did not match the code: the default `auc_horizon`, the row indices returned
  by `generate_partitioned`, automatic choice of generating features, the joint-threshold
  interaction, the logit calibration bracket, the checks on sampled outcomes, `rank_stability`
  with more than two rankings and the failure counts of `compare_paired_replicates`.
- The gallery's calibration table (`outcome-calibration.json`) matches the output of the
  gallery script.
- The reference list cites the definition of the time-dependent ROC curve (Heagerty, Lumley
  and Pepe, 2000).

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
