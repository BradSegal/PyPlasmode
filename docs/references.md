# References

The following references describe plasmode simulation, simulation-study design and ranking
comparison methods used in PyPlasmode.

- Gadbury GL et al. Evaluating statistical methods using plasmode data sets in the age of
  massive public databases: an illustration using false discovery rates. *PLOS Genetics*
  4(6), e1000098 (2008). [doi:10.1371/journal.pgen.1000098](https://doi.org/10.1371/journal.pgen.1000098).
  Basis for combining empirical data with known simulation structure.
- Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods.
  *Statistics in Medicine* 38(11), 2074--2102 (2019).
  [doi:10.1002/sim.8086](https://doi.org/10.1002/sim.8086).
  Basis for ADEMP and Monte Carlo uncertainty reporting.
- Webber W, Moffat A, Zobel J. A similarity measure for indefinite rankings.
  *ACM Transactions on Information Systems* (2010).
  [doi:10.1145/1852102.1852106](https://doi.org/10.1145/1852102.1852106).
  Basis for extrapolated rank-biased overlap; PyPlasmode supports equal-length complete rankings.

## Numerical libraries

- [NumPy random generators](https://numpy.org/doc/stable/reference/random/index.html)
  supply independent seeded streams and random sampling.
- [SciPy brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)
  solves marginal-risk calibration equations. SciPy also supplies hypergeometric references,
  Student t intervals and Spearman correlation.
- [scikit-learn metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html)
  supply prediction metrics. Its imputer and linear regression implement held-out
  signal reconstruction.
- [statsmodels](https://www.statsmodels.org/stable/index.html) supplies independent reference
  estimators for the optional validation example.
