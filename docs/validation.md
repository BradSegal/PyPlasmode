# Validation

A benchmark is only as trustworthy as its simulator. If the outcomes did not follow the
settings you asked for, or recovery were scored wrongly, every comparison built on them would
mislead. The validation example checks both on synthetic data where the right answers are
known. To run it, install the optional estimators from
[PyPI](https://pypi.org/project/pyplasmode/), download the
[validation script](https://github.com/BradSegal/PyPlasmode/blob/main/examples/validation_vignette.py)
and run it:

```bash
python -m pip install "pyplasmode[validation]"
python validation_vignette.py --output validation-output
```

It writes CSV summaries, the figure below as PDF and SVG, and a JSON description of the
simulation design. The JSON follows ADEMP (aims, data-generating mechanisms, estimands, methods
and performance measures), the reporting framework for simulation studies described by
[Morris et al.](references.md#simulation-studies).

## Reference run

The run targets three ways a simulator can fail: generating the wrong outcome, scoring a
ranking wrongly, and finding apparent signal where there is none. It proceeds in six steps:

1. **Create a population** of 20,000 rows and 100 independent Gaussian features. Because the
   features are independent and identically distributed, none has an advantage in being
   selected, so the expected result of random selection is known exactly.
2. **Add a known effect.** One feature, standardised, generates a binary, a continuous, a
   count, an ordinal and a survival outcome, each with a stated effect size.
3. **Recover the effect with another implementation.** statsmodels fits a binomial, linear,
   Poisson, ordered-logistic and Cox model to the respective outcome. If PyPlasmode generates
   outcomes correctly, each fitted coefficient's 95% interval should contain the effect it was
   given.
4. **Repeat outcome generation** with 40 random seeds per outcome family, and compare the
   realised outcomes with the requested probabilities, rates and other targets in the outcome's
   own units.
5. **Check the recovery arithmetic.** Five features generate the signal. A ranking that puts
   them first should have a top-ten recovery of 1. Random rankings should average 10 of 100, a
   recovery of 0.1; the run draws 1,000 of them.
6. **Remove the association.** In 400 simulations, set the odds ratio to one, so the binary
   outcome is unrelated to every feature, and rank the features by their association with it.
   Each feature should then reach the top ten in 400 × 10/100 = 40 simulations on average,
   and no feature should stand out.

![Effect estimates from independent implementations and feature-selection frequency under the null](assets/validation/validation.svg)

The points and 95% intervals show, for each outcome family, the statsmodels estimate minus the
effect PyPlasmode was given, from one simulated dataset each. Zero means exact agreement.
Ratio effects, such as odds ratios, are compared on the log scale. The histogram shows how many
of the 400 no-association simulations put each feature in its top ten. The dashed lines mark
the expected 40 plus and minus three standard deviations (22 and 58).

The results:

- All five fitted intervals contained the effect PyPlasmode was given.
- All 40 replicates in each outcome family met their targets without a diagnostic failure.
- The ranking with the generating features first had recovery 1. The 1,000 random rankings
  averaged 0.1018 against the expected 0.1, with a Monte Carlo standard error of 0.0042.
- Without an association, each feature reached the top ten between 23 and 54 times, all within
  three standard deviations of the expected 40.

The underlying tables are available: [effect estimates](assets/validation/external-conformance.csv),
[outcome targets in native units](assets/validation/outcome-reproducibility.csv),
[ranking controls](assets/validation/truth-recovery.csv) and
[no-association counts](assets/validation/null-control.csv). The script contains the sample
sizes, data-generating mechanisms and seeds that produced them.

## Test suite

The reference run checks one configuration. The test suite checks the parts of the library
separately and at their edges, so that a change that breaks one is caught before release.
Unit tests check input handling and each calculation. Statistical tests check that simulated
distributions and chance expectations match theory. Integration tests run sampling,
generation and evaluation together for each outcome family, including null signals and
missing data.

Each component is checked against an independent calculation, and against the cases most
likely to break it:

| Component | Reference calculation | Additional cases |
| --- | --- | --- |
| Complete-row sampling | Source-row identity and empirical covariance | Replacement limits and separation of source rows |
| Signal construction | Linear algebra for sparse, correlated and distributed signals | Missing or constant features and invalid specifications |
| Binary and survival effects | statsmodels Logit and PHReg | Infeasible probabilities and invalid cumulative risks |
| Continuous, ordinal and count effects | Marginal moments and category probabilities | Null signals, categories and varying exposures |
| Count overdispersion (NB1 negative binomial) | SciPy negative-binomial moments | Conditional residual variance with heterogeneous means |
| Exact recovery | Hand counts and SciPy hypergeometric expectations | Agreement between rankings and tied-score selections |
| Fractional group and interaction recovery | Combinatorial and sampled references | Ties, zero scores and invariant references |
| Stability and reconstruction | Prefix-overlap formula and standard regression | Small samples and constant targets |

Two further sets of tests cover recovery above chance. Matched recovery tests compare the
analytic chance expectation with hypergeometric and sampled references. Outcome specificity
tests score one selection against its own signal and against other signals on different
features, the comparison that shows whether a method used the outcome or favoured features that
are easy to select for other reasons.

The `pyplasmode.metrics` tests check:

- selection stability and rank overlap at their chance levels when rankings are exchangeable;
- exact enumeration of ties at the selection boundary;
- consensus rankings built from noisy refits;
- weighted pairwise AUC calculations;
- that the censored AUC estimator recovers the uncensored AUC under independent censoring;
- shared bootstrap intervals and the gap between a model's AUC and the generating score's.

Target AUC calibration is tested in 12 binary and 36 survival settings; the
[calibration guide](calibration.md) lists their sample sizes and tolerances. The
[metrics and calibration example](https://github.com/BradSegal/PyPlasmode/blob/main/examples/metrics_calibration.py)
runs these functions together on synthetic scores.

To run the tests yourself, follow the
[development setup](https://github.com/BradSegal/PyPlasmode/blob/main/CONTRIBUTING.md) and run
`pytest`.
