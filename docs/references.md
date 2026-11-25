# References

These references describe plasmode simulation, simulation study design, ranking comparison and
selection stability. PyPlasmode's numerical checks and reference results are documented in
[Validation](validation.md).

## Plasmode simulation

- Gadbury GL, Xiang Q, Yang L, Barnes S, Page GP, Allison DB. Evaluating statistical methods using plasmode data sets in the age of
  massive public databases: an illustration using false discovery rates. *PLOS Genetics*
  4(6), e1000098 (2008). [doi:10.1371/journal.pgen.1000098](https://doi.org/10.1371/journal.pgen.1000098).
  Uses data derived from real experiments, with known simulation structure, to compare
  false-discovery methods: the conceptual basis for the plasmode approach.
- Schreck N, Slynko A, Saadati M, Benner A. Statistical plasmode simulations–Potentials,
  challenges and recommendations. *Statistics in Medicine* 43(9), 1804-1825 (2024).
  [doi:10.1002/sim.10012](https://doi.org/10.1002/sim.10012).
  Describes resampling covariates and generating outcomes from a specified model, including
  choices of source data and simulation design: the closer match to PyPlasmode's
  generation workflow.

## Simulation studies

- Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods.
  *Statistics in Medicine* 38(11), 2074-2102 (2019).
  [doi:10.1002/sim.8086](https://doi.org/10.1002/sim.8086).
  Section 3 introduces ADEMP; Section 5.2 explains Monte Carlo uncertainty. The paper also
  distinguishes prediction, model selection and parameter estimation as simulation targets.

## Ranking comparison

- Webber W, Moffat A, Zobel J. A similarity measure for indefinite rankings.
  *ACM Transactions on Information Systems* 28(4), Article 20, 38 pp. (2010).
  [doi:10.1145/1852102.1852106](https://doi.org/10.1145/1852102.1852106).
  Equation 23 gives the extrapolated rank-biased overlap, which PyPlasmode applies to
  complete, equal-length rankings, and Equation 28 the treatment of ties. The equation
  numbers are those of the
  [author manuscript](https://www.codalism.com/research/papers/wmz10_tois.pdf), which is
  paginated differently from the published article.

## Selection stability and censored discrimination

- Nogueira S, Sechidis K, Brown G. On the Stability of Feature Selection Algorithms.
  *Journal of Machine Learning Research* 18(174), 1–54 (2018).
  [Article](https://jmlr.org/papers/v18/17-514.html). Definition 4 supplies
  chance-corrected stability for binary selections. PyPlasmode extends the measure by
  integrating independent binary resolutions of ties.
- Heagerty PJ, Lumley T, Pepe MS. Time-dependent ROC curves for censored survival data and
  a diagnostic marker. *Biometrics* 56(2), 337-344 (2000).
  [doi:10.1111/j.0006-341X.2000.00337.x](https://doi.org/10.1111/j.0006-341X.2000.00337.x).
  Defines the ROC curve at a time *t*: cases have had the event by *t* and controls are
  event-free at *t*.
- [scikit-survival cumulative/dynamic AUC](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.cumulative_dynamic_auc.html)
  documents the estimator with inverse-probability-of-censoring weights that
  `cumulative_dynamic_auc` implements.

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
  compute AUC, average precision, log loss, R-squared and root mean squared error.
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
