"""Monte Carlo calibration checks for native-unit outcome generation."""

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg

from pyplasmode import (
    BinaryOutcome,
    ContinuousOutcome,
    CountOutcome,
    OrdinalOutcome,
    TimeToEventOutcome,
    generate_outcome,
    years,
)
from pyplasmode.outcomes import (
    BinaryGeneratedOutcome,
    ContinuousGeneratedOutcome,
    CountGeneratedOutcome,
    OrdinalGeneratedOutcome,
    SurvivalGeneratedOutcome,
)


def _signal(size: int = 60_000) -> np.ndarray:
    values = np.random.default_rng(101).normal(size=size)
    return (values - values.mean()) / values.std()


@pytest.mark.statistical
@pytest.mark.parametrize(
    "spec",
    [
        BinaryOutcome(0.12, odds_ratio=1.8),
        BinaryOutcome(0.12, risk_ratio=1.2),
        BinaryOutcome(0.12, risk_difference=0.02),
    ],
)
def test_binary_targets_recur_in_native_probability_units(spec: BinaryOutcome) -> None:
    result = generate_outcome(spec, _signal(), np.random.default_rng(8))
    assert isinstance(result, BinaryGeneratedOutcome)
    assert abs(result.values.mean() - 0.12) < 0.01
    assert all(check.within_tolerance for check in result.calibration.checks)


@pytest.mark.statistical
def test_continuous_mean_spread_and_variance_target_recur() -> None:
    result = generate_outcome(
        ContinuousOutcome(10.0, 2.0, variance_explained=0.25),
        _signal(),
        np.random.default_rng(9),
    )
    assert isinstance(result, ContinuousGeneratedOutcome)
    assert abs(result.values.mean() - 10.0) < 0.05
    assert abs(result.values.std() - 2.0) < 0.05


@pytest.mark.statistical
@pytest.mark.parametrize("variance_to_mean", [1.0, 2.0])
def test_count_rate_recurs_for_poisson_and_overdispersed_counts(
    variance_to_mean: float,
) -> None:
    result = generate_outcome(
        CountOutcome(0.4, 1.4, exposure=2.0, variance_to_mean=variance_to_mean),
        _signal(),
        np.random.default_rng(10),
    )
    assert isinstance(result, CountGeneratedOutcome)
    assert abs(result.values.sum() / (2.0 * result.values.size) - 0.4) < 0.015


@pytest.mark.statistical
@pytest.mark.parametrize("exposure", [0.5, 1.0, 10.0])
def test_count_conditional_dispersion_matches_scipy_at_each_exposure(exposure: float) -> None:
    """NB1 count variance scales with the supplied mean, including exposure."""
    from scipy.stats import nbinom

    ratio = 2.0
    mean = 1.0 * exposure
    reference_mean, reference_variance = nbinom.stats(
        mean / (ratio - 1.0), 1.0 / ratio, moments="mv"
    )
    result = generate_outcome(
        CountOutcome(1.0, 1.0, exposure=exposure, variance_to_mean=ratio),
        np.zeros(100_000),
        np.random.default_rng(103),
    )
    assert isinstance(result, CountGeneratedOutcome)
    assert result.values.mean() == pytest.approx(float(reference_mean), rel=0.03)
    assert result.values.var() == pytest.approx(float(reference_variance), rel=0.06)
    assert result.values.var() / result.values.mean() == pytest.approx(ratio, rel=0.05)


@pytest.mark.statistical
def test_count_dispersion_is_conditional_under_heterogeneous_means() -> None:
    signal = _signal(100_000)
    exposure = np.linspace(0.5, 3.0, signal.size)
    result = generate_outcome(
        CountOutcome(1.0, 1.5, exposure=tuple(exposure), variance_to_mean=2.0),
        signal,
        np.random.default_rng(104),
    )
    assert isinstance(result, CountGeneratedOutcome)
    parameters = dict(result.calibration.parameters)
    means = exposure * np.exp(float(parameters["intercept"]) + np.log(1.5) * signal)
    pearson_variance = np.mean((result.values - means) ** 2 / means)
    assert pearson_variance == pytest.approx(2.0, rel=0.05)


@pytest.mark.statistical
def test_ordinal_category_probabilities_recur() -> None:
    result = generate_outcome(
        OrdinalOutcome(("low", "middle", "high"), (0.2, 0.5, 0.3), 1.5),
        _signal(),
        np.random.default_rng(11),
    )
    assert isinstance(result, OrdinalGeneratedOutcome)
    realised = np.bincount(result.codes, minlength=3) / result.codes.size
    np.testing.assert_allclose(realised, (0.2, 0.5, 0.3), atol=0.01)


@pytest.mark.statistical
def test_piecewise_survival_risk_and_censoring_targets_recur() -> None:
    horizons = {years(5): 0.08, years(10): 0.18}
    result = generate_outcome(
        TimeToEventOutcome(
            horizons,
            hazard_ratio=1.6,
            censoring_risks={years(5): 0.03, years(10): 0.08},
        ),
        _signal(),
        np.random.default_rng(12),
    )
    assert isinstance(result, SurvivalGeneratedOutcome)
    for horizon, target in horizons.items():
        assert abs(np.mean(result.latent_event_time <= horizon.days) - target) < 0.01
    assert abs(np.mean(result.censoring_time <= years(10).days) - 0.08) < 0.01
    hazards = dict(result.calibration.parameters)["baseline_interval_hazards"]
    assert all(value >= 0.0 for value in hazards)


@pytest.mark.statistical
def test_binary_odds_ratio_matches_statsmodels_logit_reference() -> None:
    """Native binary calibration reproduces its effect in a maintained estimator."""
    signal = _signal(30_000)
    result = generate_outcome(
        BinaryOutcome(0.12, odds_ratio=1.8),
        signal,
        np.random.default_rng(108),
    )
    assert isinstance(result, BinaryGeneratedOutcome)

    fitted = sm.Logit(result.values, sm.add_constant(signal)).fit(disp=False)

    assert np.exp(fitted.params[1]) == pytest.approx(1.8, rel=0.10)


@pytest.mark.statistical
def test_survival_hazard_ratio_matches_statsmodels_phreg_reference() -> None:
    """Generated proportional hazards recover in an independent maintained fit."""
    signal = _signal(30_000)
    result = generate_outcome(
        TimeToEventOutcome(
            {years(10): 0.18},
            hazard_ratio=1.6,
            censoring_risks={years(10): 0.08},
        ),
        signal,
        np.random.default_rng(112),
    )
    assert isinstance(result, SurvivalGeneratedOutcome)

    fitted = PHReg(result.time, signal[:, None], status=result.event).fit(disp=False)

    assert np.exp(fitted.params[0]) == pytest.approx(1.6, rel=0.10)
