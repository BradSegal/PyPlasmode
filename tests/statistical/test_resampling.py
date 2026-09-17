"""Statistical checks for empirical complete-row resampling."""

import numpy as np
import pytest

from pyplasmode import Population, resample_population


@pytest.mark.statistical
def test_resampling_preserves_joint_covariance_and_missingness_within_sampling_error() -> None:
    rng = np.random.default_rng(11)
    source = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.75], [0.75, 1.0]], size=2_000)
    source[source[:, 0] > 1.5, 1] = np.nan
    population = Population(source, ("a", "b"))
    sample = resample_population(population, 20_000, np.random.default_rng(12))

    source_complete = source[~np.isnan(source).any(axis=1)]
    sample_complete = sample.X[~np.isnan(sample.X).any(axis=1)]
    assert abs(np.corrcoef(source_complete.T)[0, 1] - np.corrcoef(sample_complete.T)[0, 1]) < 0.03
    assert abs(np.isnan(source[:, 1]).mean() - np.isnan(sample.X[:, 1]).mean()) < 0.01
