# References

These sources describe plasmode simulation, simulation-study design and ranking comparison.
PyPlasmode's own numerical checks and reference results are documented in [Validation](validation.md).

## Plasmode simulation

- Gadbury GL, Xiang Q, Yang L, Barnes S, Page GP, Allison DB. Evaluating statistical methods using plasmode data sets in the age of
  massive public databases: an illustration using false discovery rates. *PLOS Genetics*
  4(6), e1000098 (2008). [doi:10.1371/journal.pgen.1000098](https://doi.org/10.1371/journal.pgen.1000098).
  Uses data derived from real experiments, with known simulation structure, to compare
  false-discovery methods. This is the conceptual basis for the plasmode approach.
- Schreck N, Slynko A, Saadati M, Benner A. Statistical plasmode simulations: Potentials,
  challenges and recommendations. *Statistics in Medicine* 43(9), 1804-1825 (2024).
  [doi:10.1002/sim.10012](https://doi.org/10.1002/sim.10012).
  Describes resampling covariates and generating outcomes from a specified model, including
  choices of source data and simulation design. This more closely matches PyPlasmode's
  generation workflow.

## Simulation studies

- Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods.
  *Statistics in Medicine* 38(11), 2074-2102 (2019).
  [doi:10.1002/sim.8086](https://doi.org/10.1002/sim.8086).
  Section 3 introduces ADEMP; Section 5.2 explains Monte Carlo uncertainty. The paper also
  distinguishes prediction, model selection and parameter estimation as simulation targets.

## Ranking comparison

- Webber W, Moffat A, Zobel J. A similarity measure for indefinite rankings.
  *ACM Transactions on Information Systems* 28(4), 1-38 (2010).
  [doi:10.1145/1852102.1852106](https://doi.org/10.1145/1852102.1852106).
  Section 4.4, Equation 23 gives the extrapolated rank-biased overlap used by PyPlasmode.
  PyPlasmode applies it to complete, equal-length rankings. An
  [author manuscript](https://codalism.com/research/papers/wmz10_tois.pdf) is available;
  its pagination differs from the published article.

## Numerical libraries

- **Random sampling:** [NumPy generators](https://numpy.org/doc/stable/reference/random/index.html)
  draw samples; [SeedSequence spawning](https://numpy.org/doc/stable/reference/random/parallel.html)
  creates separate seeded streams for sampling, signal construction and outcome generation.
- **Outcome calibration:** [SciPy brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)
  solves the expectation equations defined in [Outcomes](outcomes.md#calibration-equations).
- **Chance recovery:** [SciPy hypergeom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html)
  supplies probabilities for sampling without replacement, used in matched recovery and its tests.
- **Count dispersion:** [SciPy nbinom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html)
  supplies reference moments for the negative-binomial generator tests.
- **Ranking and uncertainty:** [SciPy spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
  compares rankings; [Student's t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)
  supplies the critical value for replicate-mean intervals.
- **Prediction:** [scikit-learn metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html)
  compute ROC AUC, average precision, log loss, R-squared and root mean squared error.
- **Signal reconstruction:** scikit-learn's
  [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
  and [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
  fit median imputation and ordinary least squares on development rows.
- **Tutorial tuning:** scikit-learn's
  [pipeline guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
  explains why preprocessing is fitted within cross-validation folds.
- **Reference effect estimates:** statsmodels
  [GLM](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html),
  [OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html),
  [OrderedModel](https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html)
  and [PHReg](https://www.statsmodels.org/stable/generated/statsmodels.duration.hazard_regression.PHReg.html)
  provide the binomial, Poisson, linear, ordered-logistic and Cox fits in the
  [validation example](validation.md).
