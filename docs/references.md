# References

## Plasmode simulation

- Gadbury GL, Xiang Q, Yang L, Barnes S, Page GP, Allison DB. Evaluating statistical methods
  using plasmode data sets in the age of massive public databases: an illustration using false
  discovery rates. *PLOS Genetics* 4(6), e1000098 (2008).
  [doi:10.1371/journal.pgen.1000098](https://doi.org/10.1371/journal.pgen.1000098).
- Schreck N, Slynko A, Saadati M, Benner A. Statistical plasmode simulations–Potentials,
  challenges and recommendations. *Statistics in Medicine* 43(9), 1804–1825 (2024).
  [doi:10.1002/sim.10012](https://doi.org/10.1002/sim.10012).

## Simulation studies

- Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods.
  *Statistics in Medicine* 38(11), 2074–2102 (2019).
  [doi:10.1002/sim.8086](https://doi.org/10.1002/sim.8086).

## Ranking comparison

- Webber W, Moffat A, Zobel J. A similarity measure for indefinite rankings.
  *ACM Transactions on Information Systems* 28(4), Article 20 (2010).
  [doi:10.1145/1852102.1852106](https://doi.org/10.1145/1852102.1852106).
  Equation numbers follow the
  [author manuscript](https://www.codalism.com/research/papers/wmz10_tois.pdf).

## Selection stability and censored discrimination

- Nogueira S, Sechidis K, Brown G. On the stability of feature selection algorithms.
  *Journal of Machine Learning Research* 18(174), 1–54 (2018).
  [jmlr.org/papers/v18/17-514](https://jmlr.org/papers/v18/17-514.html).
- Heagerty PJ, Lumley T, Pepe MS. Time-dependent ROC curves for censored survival data and
  a diagnostic marker. *Biometrics* 56(2), 337–344 (2000).
  [doi:10.1111/j.0006-341X.2000.00337.x](https://doi.org/10.1111/j.0006-341X.2000.00337.x).

## Software

PyPlasmode is built on [NumPy](https://numpy.org/doc/stable/reference/random/index.html),
[SciPy](https://docs.scipy.org/doc/scipy/reference/) and
[scikit-learn](https://scikit-learn.org/stable/api/sklearn.metrics.html). The
[validation example](validation.md) compares its results with
[statsmodels](https://www.statsmodels.org/stable/), and `cumulative_dynamic_auc` implements the
estimator documented by
[scikit-survival](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.cumulative_dynamic_auc.html).
