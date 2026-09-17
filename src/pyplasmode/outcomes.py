"""Calibrated native-unit outcome specifications and generators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import expit

type ParameterValue = float | str | tuple[float, ...]


class OutcomeSpecificationError(ValueError):
    """Raised when observable outcome targets are invalid or ambiguous."""


class CalibrationError(RuntimeError):
    """Raised when valid targets cannot be represented by the documented model."""


class RealizationError(RuntimeError):
    """Raised when a generated sample exceeds its sampling-error tolerance."""


@dataclass(frozen=True, order=True, slots=True)
class Duration:
    """A positive native-time duration resolved to days."""

    days: float
    label: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.days) or self.days <= 0.0:
            raise OutcomeSpecificationError("duration must be finite and positive")
        if not self.label:
            raise OutcomeSpecificationError("duration label must be non-empty")


def duration(value: float, unit: Literal["days", "weeks", "years"]) -> Duration:
    """Create a positive duration in days, weeks or years."""
    factors = {"days": 1.0, "weeks": 7.0, "years": 365.25}
    if not np.isfinite(value) or value <= 0.0:
        raise OutcomeSpecificationError("duration value must be finite and positive")
    if unit not in factors:
        raise OutcomeSpecificationError("duration unit must be days, weeks or years")
    return Duration(float(value) * factors[unit], f"{value:g} {unit}")


def days(value: float) -> Duration:
    """Create a duration measured in days."""
    return duration(value, "days")


def years(value: float) -> Duration:
    """Create a duration measured in years using 365.25 days per year."""
    return duration(value, "years")


@dataclass(frozen=True, slots=True)
class BinaryOutcome:
    """A marginal binary probability and one effect per signal standard deviation."""

    probability: float
    odds_ratio: float | None = None
    risk_ratio: float | None = None
    risk_difference: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.probability < 1.0:
            raise OutcomeSpecificationError("binary probability must be between zero and one")
        effects = (self.odds_ratio, self.risk_ratio, self.risk_difference)
        if sum(effect is not None for effect in effects) != 1:
            raise OutcomeSpecificationError("BinaryOutcome requires exactly one effect measure")
        for ratio in (self.odds_ratio, self.risk_ratio):
            if ratio is not None and (not np.isfinite(ratio) or ratio <= 0.0):
                raise OutcomeSpecificationError("binary ratio effects must be finite and positive")
        if self.risk_difference is not None and not np.isfinite(self.risk_difference):
            raise OutcomeSpecificationError("risk_difference must be finite")


@dataclass(frozen=True, slots=True)
class ContinuousOutcome:
    """A marginal mean and spread with an effect or variance-explained target."""

    mean: float
    standard_deviation: float
    effect: float | None = None
    variance_explained: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise OutcomeSpecificationError("continuous mean must be finite")
        if not np.isfinite(self.standard_deviation) or self.standard_deviation <= 0.0:
            raise OutcomeSpecificationError("continuous standard_deviation must be positive")
        if (self.effect is None) == (self.variance_explained is None):
            raise OutcomeSpecificationError(
                "ContinuousOutcome requires exactly one of effect or variance_explained"
            )
        if self.effect is not None:
            if not np.isfinite(self.effect):
                raise OutcomeSpecificationError("continuous effect must be finite")
            if abs(self.effect) >= self.standard_deviation:
                raise OutcomeSpecificationError(
                    "continuous effect variance must be smaller than total variance"
                )
        if self.variance_explained is not None and not 0.0 <= self.variance_explained < 1.0:
            raise OutcomeSpecificationError("variance_explained must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class CountOutcome:
    """A rate per exposure and a conditional count variance-to-mean ratio."""

    rate: float
    rate_ratio: float
    exposure: float | tuple[float, ...] = 1.0
    variance_to_mean: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.rate) or self.rate <= 0.0:
            raise OutcomeSpecificationError("count rate must be finite and positive")
        if not np.isfinite(self.rate_ratio) or self.rate_ratio <= 0.0:
            raise OutcomeSpecificationError("rate_ratio must be finite and positive")
        exposure = np.asarray(self.exposure, dtype=float)
        if exposure.ndim > 1 or not exposure.size or not np.isfinite(exposure).all():
            raise OutcomeSpecificationError("exposure must be a finite scalar or vector")
        if np.any(exposure <= 0.0):
            raise OutcomeSpecificationError("exposure must be positive")
        if not np.isfinite(self.variance_to_mean) or self.variance_to_mean < 1.0:
            raise OutcomeSpecificationError("variance_to_mean must be at least one")


@dataclass(frozen=True, slots=True)
class OrdinalOutcome:
    """Ordered category probabilities with one common odds ratio."""

    categories: tuple[str, ...]
    probabilities: tuple[float, ...]
    common_odds_ratio: float

    def __post_init__(self) -> None:
        if len(self.categories) < 2 or len(set(self.categories)) != len(self.categories):
            raise OutcomeSpecificationError("ordinal categories must be ordered and unique")
        if len(self.probabilities) != len(self.categories):
            raise OutcomeSpecificationError("ordinal probabilities require one value per category")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.probabilities):
            raise OutcomeSpecificationError("ordinal probabilities must be finite and positive")
        if not np.isclose(sum(self.probabilities), 1.0, atol=1e-10):
            raise OutcomeSpecificationError("ordinal probabilities must sum to one")
        if not np.isfinite(self.common_odds_ratio) or self.common_odds_ratio <= 0.0:
            raise OutcomeSpecificationError("common_odds_ratio must be finite and positive")


@dataclass(frozen=True, slots=True)
class RiskPoint:
    """One cumulative risk at a native-time horizon."""

    horizon: Duration
    probability: float


def _risk_points(values: Mapping[Duration, float], *, name: str) -> tuple[RiskPoint, ...]:
    if not values:
        raise OutcomeSpecificationError(f"{name} must contain at least one horizon")
    points = tuple(
        RiskPoint(horizon, float(probability)) for horizon, probability in values.items()
    )
    points = tuple(sorted(points, key=lambda point: point.horizon.days))
    probabilities = tuple(point.probability for point in points)
    if any(not 0.0 <= value < 1.0 for value in probabilities):
        raise OutcomeSpecificationError(f"{name} probabilities must be in [0, 1)")
    if any(right < left for left, right in pairwise(probabilities)):
        raise OutcomeSpecificationError(f"{name} probabilities must be nondecreasing")
    horizons = tuple(point.horizon.days for point in points)
    if len(set(horizons)) != len(horizons):
        raise OutcomeSpecificationError(f"{name} horizons must be unique")
    return points


@dataclass(frozen=True, slots=True, init=False)
class TimeToEventOutcome:
    """Cumulative event/censoring risks and a proportional hazard ratio."""

    event_risks: tuple[RiskPoint, ...]
    hazard_ratio: float
    censoring_risks: tuple[RiskPoint, ...]

    def __init__(
        self,
        event_risks: Mapping[Duration, float],
        hazard_ratio: float,
        censoring_risks: Mapping[Duration, float] | None = None,
    ) -> None:
        events = _risk_points(event_risks, name="event_risks")
        if not np.isfinite(hazard_ratio) or hazard_ratio <= 0.0:
            raise OutcomeSpecificationError("hazard_ratio must be finite and positive")
        censoring = (
            _risk_points(censoring_risks, name="censoring_risks")
            if censoring_risks is not None
            else tuple(RiskPoint(point.horizon, 0.0) for point in events)
        )
        if tuple(point.horizon.days for point in censoring) != tuple(
            point.horizon.days for point in events
        ):
            raise OutcomeSpecificationError(
                "censoring_risks must use the same horizons as event_risks"
            )
        object.__setattr__(self, "event_risks", events)
        object.__setattr__(self, "hazard_ratio", float(hazard_ratio))
        object.__setattr__(self, "censoring_risks", censoring)


type OutcomeSpec = (
    BinaryOutcome | ContinuousOutcome | CountOutcome | OrdinalOutcome | TimeToEventOutcome
)


@dataclass(frozen=True, slots=True)
class CalibrationCheck:
    """One target, expectation, realization and sampling-error tolerance."""

    name: str
    target: float
    expected: float
    realized: float
    tolerance: float
    within_tolerance: bool


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Outcome calibration family, solved parameters and native-unit checks."""

    family: str
    parameters: tuple[tuple[str, ParameterValue], ...]
    checks: tuple[CalibrationCheck, ...]


def _readonly(values: NDArray[np.generic]) -> NDArray[np.generic]:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class BinaryGeneratedOutcome:
    """Generated binary labels and calibration."""

    values: NDArray[np.int8]
    calibration: CalibrationResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly(self.values))


@dataclass(frozen=True, slots=True)
class ContinuousGeneratedOutcome:
    """Generated continuous values and calibration."""

    values: NDArray[np.float64]
    calibration: CalibrationResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly(self.values))


@dataclass(frozen=True, slots=True)
class CountGeneratedOutcome:
    """Generated counts, exposure and calibration."""

    values: NDArray[np.int64]
    exposure: NDArray[np.float64]
    calibration: CalibrationResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly(self.values))
        object.__setattr__(self, "exposure", _readonly(self.exposure))


@dataclass(frozen=True, slots=True)
class OrdinalGeneratedOutcome:
    """Generated ordinal codes, category labels and calibration."""

    codes: NDArray[np.int64]
    categories: tuple[str, ...]
    calibration: CalibrationResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "codes", _readonly(self.codes))


@dataclass(frozen=True, slots=True)
class SurvivalGeneratedOutcome:
    """Observed follow-up, event state and latent generation times."""

    time: NDArray[np.float64]
    event: NDArray[np.bool_]
    latent_event_time: NDArray[np.float64]
    censoring_time: NDArray[np.float64]
    calibration: CalibrationResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _readonly(self.time))
        object.__setattr__(self, "event", _readonly(self.event))
        object.__setattr__(self, "latent_event_time", _readonly(self.latent_event_time))
        object.__setattr__(self, "censoring_time", _readonly(self.censoring_time))


type GeneratedOutcome = (
    BinaryGeneratedOutcome
    | ContinuousGeneratedOutcome
    | CountGeneratedOutcome
    | OrdinalGeneratedOutcome
    | SurvivalGeneratedOutcome
)


def _signal(values: NDArray[np.float64]) -> NDArray[np.float64]:
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
        raise CalibrationError("signal must be a finite non-empty one-dimensional array")
    spread = float(np.std(signal))
    if spread == 0.0:
        return np.zeros_like(signal)
    return (signal - np.mean(signal)) / spread


def _probability_tolerance(target: float, size: int) -> float:
    return float(max(6.0 * np.sqrt(target * (1.0 - target) / size), 6.0 / size))


def _check(
    name: str, target: float, expected: float, realized: float, tolerance: float
) -> CalibrationCheck:
    if not np.isclose(expected, target, rtol=1e-8, atol=1e-10):
        raise CalibrationError(f"{name} expectation {expected:g} does not match target {target:g}")
    within = abs(realized - target) <= tolerance
    if not within:
        raise RealizationError(
            f"{name} realization {realized:g} differs from target {target:g} "
            f"beyond tolerance {tolerance:g}"
        )
    return CalibrationCheck(name, target, expected, realized, tolerance, True)


def _logit_intercept(offset: NDArray[np.float64], target: float) -> float:
    """Solve a marginal probability without silently widening the supported model."""
    try:
        return float(
            brentq(lambda value: float(np.mean(expit(value + offset)) - target), -50.0, 50.0)
        )
    except ValueError as error:
        raise CalibrationError(
            "requested probability is outside the logit calibration bracket"
        ) from error


def _solve_monotone(function: object, target_name: str) -> float:
    if not callable(function):
        raise TypeError("calibration function must be callable")
    upper = 1.0
    while function(upper) > 0.0 and upper < 1e6:
        upper *= 2.0
    try:
        return float(brentq(function, 0.0, upper, xtol=1e-12, rtol=1e-12))
    except ValueError as error:
        raise CalibrationError(f"could not bracket {target_name} calibration") from error


def _generate_binary(
    spec: BinaryOutcome, z: NDArray[np.float64], rng: np.random.Generator
) -> BinaryGeneratedOutcome:
    if spec.odds_ratio is not None:
        beta = float(np.log(spec.odds_ratio))
        intercept = _logit_intercept(beta * z, spec.probability)
        probabilities = expit(intercept + beta * z)
        family = "bernoulli_logit"
        effect_name = "odds_ratio"
        effect_value = spec.odds_ratio
    elif spec.risk_ratio is not None:
        beta = float(np.log(spec.risk_ratio))
        intercept = float(np.log(spec.probability / np.mean(np.exp(beta * z))))
        probabilities = np.exp(intercept + beta * z)
        family = "bernoulli_log"
        effect_name = "risk_ratio"
        effect_value = spec.risk_ratio
    else:
        if spec.risk_difference is None:
            raise RuntimeError("binary effect is missing")
        beta = float(spec.risk_difference)
        intercept = float(spec.probability - beta * np.mean(z))
        probabilities = intercept + beta * z
        family = "bernoulli_identity"
        effect_name = "risk_difference"
        effect_value = spec.risk_difference
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise CalibrationError("binary conditional probabilities leave [0, 1]")
    values = rng.binomial(1, probabilities).astype(np.int8)
    expected = float(np.mean(probabilities))
    realized = float(np.mean(values))
    check = _check(
        "probability",
        spec.probability,
        expected,
        realized,
        _probability_tolerance(spec.probability, values.size),
    )
    calibration = CalibrationResult(
        family,
        (("intercept", intercept), ("signal_coefficient", beta), (effect_name, effect_value)),
        (check,),
    )
    return BinaryGeneratedOutcome(values, calibration)


def _generate_continuous(
    spec: ContinuousOutcome, z: NDArray[np.float64], rng: np.random.Generator
) -> ContinuousGeneratedOutcome:
    if np.ptp(z) == 0.0 and (
        spec.effect not in (None, 0.0) or spec.variance_explained not in (None, 0.0)
    ):
        raise OutcomeSpecificationError(
            "a constant signal cannot explain continuous outcome variance"
        )
    if z.size < 2:
        raise OutcomeSpecificationError("continuous generation requires at least two observations")
    if spec.effect is not None:
        effect = float(spec.effect)
        residual_sd = float(np.sqrt(spec.standard_deviation**2 - effect**2))
        target_r2 = (effect / spec.standard_deviation) ** 2
    else:
        if spec.variance_explained is None:
            raise RuntimeError("continuous effect is missing")
        target_r2 = float(spec.variance_explained)
        effect = float(spec.standard_deviation * np.sqrt(target_r2))
        residual_sd = float(spec.standard_deviation * np.sqrt(1.0 - target_r2))
    values = spec.mean + effect * z + rng.normal(0.0, residual_sd, size=z.size)
    mean_tolerance = 6.0 * spec.standard_deviation / np.sqrt(z.size)
    sd_tolerance = 6.0 * spec.standard_deviation / np.sqrt(2.0 * (z.size - 1))
    checks = (
        _check("mean", spec.mean, spec.mean, float(np.mean(values)), mean_tolerance),
        _check(
            "standard_deviation",
            spec.standard_deviation,
            spec.standard_deviation,
            float(np.std(values)),
            sd_tolerance,
        ),
    )
    calibration = CalibrationResult(
        "gaussian",
        (
            ("intercept", spec.mean),
            ("signal_coefficient", effect),
            ("residual_standard_deviation", residual_sd),
            ("variance_explained", target_r2),
        ),
        checks,
    )
    return ContinuousGeneratedOutcome(np.asarray(values, dtype=np.float64), calibration)


def _exposure(spec: CountOutcome, size: int) -> NDArray[np.float64]:
    values = np.asarray(spec.exposure, dtype=np.float64)
    if values.ndim == 0:
        return np.full(size, float(values), dtype=np.float64)
    if values.shape != (size,):
        raise OutcomeSpecificationError("exposure vector length must equal signal length")
    return values


def _generate_count(
    spec: CountOutcome, z: NDArray[np.float64], rng: np.random.Generator
) -> CountGeneratedOutcome:
    exposure = _exposure(spec, z.size)
    beta = float(np.log(spec.rate_ratio))
    multiplier = np.exp(beta * z)
    intercept = float(np.log(spec.rate * np.sum(exposure) / np.sum(exposure * multiplier)))
    means = exposure * np.exp(intercept + beta * z)
    if spec.variance_to_mean == 1.0:
        values = rng.poisson(means).astype(np.int64)
        dispersion = 0.0
        variance = means
        family = "poisson_log"
    else:
        dispersion = float(spec.variance_to_mean - 1.0)
        shape = means / dispersion
        latent_means = rng.gamma(shape=shape, scale=dispersion)
        values = rng.poisson(latent_means).astype(np.int64)
        variance = spec.variance_to_mean * means
        family = "negative_binomial_nb1_log"
    expected_rate = float(np.sum(means) / np.sum(exposure))
    realized_rate = float(np.sum(values) / np.sum(exposure))
    tolerance = float(6.0 * np.sqrt(np.sum(variance)) / np.sum(exposure))
    check = _check("rate", spec.rate, expected_rate, realized_rate, tolerance)
    calibration = CalibrationResult(
        family,
        (
            ("intercept", intercept),
            ("signal_coefficient", beta),
            ("rate_ratio", spec.rate_ratio),
            ("negative_binomial_dispersion", dispersion),
            ("conditional_variance_to_mean", spec.variance_to_mean),
        ),
        (check,),
    )
    return CountGeneratedOutcome(values, exposure, calibration)


def _generate_ordinal(
    spec: OrdinalOutcome, z: NDArray[np.float64], rng: np.random.Generator
) -> OrdinalGeneratedOutcome:
    beta = float(np.log(spec.common_odds_ratio))
    cumulative_targets = np.cumsum(spec.probabilities)[:-1]
    thresholds = tuple(_logit_intercept(-beta * z, float(target)) for target in cumulative_targets)
    if any(right <= left for left, right in pairwise(thresholds)):
        raise CalibrationError("calibrated ordinal thresholds are not strictly ordered")
    conditional_cdf = np.column_stack([expit(value - beta * z) for value in thresholds])
    uniforms = rng.random(z.size)
    codes = np.sum(uniforms[:, None] > conditional_cdf, axis=1).astype(np.int64)
    expected_probabilities = np.diff(
        np.concatenate(([0.0], np.mean(conditional_cdf, axis=0), [1.0]))
    )
    realized_probabilities = np.bincount(codes, minlength=len(spec.categories)) / codes.size
    checks = tuple(
        _check(
            f"category_probability:{category}",
            target,
            float(expected),
            float(realized),
            _probability_tolerance(target, codes.size),
        )
        for category, target, expected, realized in zip(
            spec.categories,
            spec.probabilities,
            expected_probabilities,
            realized_probabilities,
            strict=True,
        )
    )
    calibration = CalibrationResult(
        "proportional_odds_logit",
        (
            ("thresholds", thresholds),
            ("signal_coefficient", beta),
            ("common_odds_ratio", spec.common_odds_ratio),
        ),
        checks,
    )
    return OrdinalGeneratedOutcome(codes, spec.categories, calibration)


def _cumulative_hazards(
    points: tuple[RiskPoint, ...], multiplier: NDArray[np.float64]
) -> NDArray[np.float64]:
    solved: list[float] = []
    for point in points:
        target_survival = 1.0 - point.probability
        solved.append(
            _solve_monotone(
                lambda value, target=target_survival: float(
                    np.mean(np.exp(-value * multiplier)) - target
                ),
                f"risk at {point.horizon.label}",
            )
        )
    hazards = np.asarray(solved, dtype=np.float64)
    if np.any(np.diff(hazards) < -1e-10):
        raise CalibrationError("calibrated cumulative event hazards are not nondecreasing")
    return hazards


def _interval_hazards(
    horizons: NDArray[np.float64], cumulative: NDArray[np.float64]
) -> NDArray[np.float64]:
    increments = np.diff(np.concatenate(([0.0], cumulative)))
    widths = np.diff(np.concatenate(([0.0], horizons)))
    rates = increments / widths
    if np.any(rates < -1e-12) or not np.isfinite(rates).all():
        raise CalibrationError("piecewise interval hazards must be finite and non-negative")
    return np.maximum(rates, 0.0)


def _invert_piecewise(
    thresholds: NDArray[np.float64],
    horizons: NDArray[np.float64],
    cumulative: NDArray[np.float64],
) -> NDArray[np.float64]:
    times = np.full(thresholds.size, np.inf, dtype=np.float64)
    previous_hazard = 0.0
    previous_time = 0.0
    for horizon, cumulative_hazard in zip(horizons, cumulative, strict=True):
        interval_rate = (cumulative_hazard - previous_hazard) / (horizon - previous_time)
        selected = np.isinf(times) & (thresholds <= cumulative_hazard)
        if interval_rate > 0.0:
            times[selected] = (
                previous_time + (thresholds[selected] - previous_hazard) / interval_rate
            )
        previous_hazard = cumulative_hazard
        previous_time = horizon
    return times


def _generate_survival(
    spec: TimeToEventOutcome, z: NDArray[np.float64], rng: np.random.Generator
) -> SurvivalGeneratedOutcome:
    beta = float(np.log(spec.hazard_ratio))
    multiplier = np.exp(beta * z)
    horizons = np.asarray([point.horizon.days for point in spec.event_risks], dtype=np.float64)
    baseline_cumulative = _cumulative_hazards(spec.event_risks, multiplier)
    baseline_intervals = _interval_hazards(horizons, baseline_cumulative)
    censoring_cumulative = np.asarray(
        [-np.log1p(-point.probability) for point in spec.censoring_risks], dtype=np.float64
    )
    censoring_intervals = _interval_hazards(horizons, censoring_cumulative)

    event_threshold = rng.exponential(size=z.size) / multiplier
    censor_threshold = rng.exponential(size=z.size)
    latent_event_time = _invert_piecewise(event_threshold, horizons, baseline_cumulative)
    censoring_time = _invert_piecewise(censor_threshold, horizons, censoring_cumulative)
    administrative = float(horizons[-1])
    event = (latent_event_time <= censoring_time) & (latent_event_time <= administrative)
    observed_time = np.minimum(np.minimum(latent_event_time, censoring_time), administrative)

    checks: list[CalibrationCheck] = []
    for index, point in enumerate(spec.event_risks):
        expected = float(1.0 - np.mean(np.exp(-baseline_cumulative[index] * multiplier)))
        realized = float(np.mean(latent_event_time <= point.horizon.days))
        checks.append(
            _check(
                f"latent_event_risk:{point.horizon.label}",
                point.probability,
                expected,
                realized,
                _probability_tolerance(point.probability, z.size),
            )
        )
    for point in spec.censoring_risks:
        expected = point.probability
        realized = float(np.mean(censoring_time <= point.horizon.days))
        checks.append(
            _check(
                f"censoring_risk:{point.horizon.label}",
                point.probability,
                expected,
                realized,
                _probability_tolerance(point.probability, z.size),
            )
        )
    calibration = CalibrationResult(
        "piecewise_exponential_proportional_hazards",
        (
            ("signal_coefficient", beta),
            ("hazard_ratio", spec.hazard_ratio),
            ("baseline_cumulative_hazards", tuple(float(value) for value in baseline_cumulative)),
            ("baseline_interval_hazards", tuple(float(value) for value in baseline_intervals)),
            ("censoring_interval_hazards", tuple(float(value) for value in censoring_intervals)),
        ),
        tuple(checks),
    )
    return SurvivalGeneratedOutcome(
        observed_time,
        np.asarray(event, dtype=np.bool_),
        latent_event_time,
        censoring_time,
        calibration,
    )


def generate_outcome(
    spec: OutcomeSpec,
    signal: NDArray[np.float64],
    rng: np.random.Generator,
) -> GeneratedOutcome:
    """Calibrate and draw an outcome from a finite one-dimensional signal.

    Args:
        spec: Native-unit outcome specification.
        signal: Finite vector, standardised internally to mean zero and unit
            population standard deviation; constant vectors represent no signal.
        rng: Caller-owned NumPy generator.

    Returns:
        Typed outcome arrays and solved calibration parameters. Survival times
        are in days and are administratively censored at the last horizon.

    Raises:
        OutcomeSpecificationError: Invalid targets, exposure dimensions, or a
            positive continuous explained-variance request with constant signal.
        CalibrationError: The chosen family cannot attain a requested target.
        RealizationError: Finite-sample diagnostics exceed the declared tolerance.
    """
    z = _signal(signal)
    if isinstance(spec, BinaryOutcome):
        return _generate_binary(spec, z, rng)
    if isinstance(spec, ContinuousOutcome):
        return _generate_continuous(spec, z, rng)
    if isinstance(spec, CountOutcome):
        return _generate_count(spec, z, rng)
    if isinstance(spec, OrdinalOutcome):
        return _generate_ordinal(spec, z, rng)
    if isinstance(spec, TimeToEventOutcome):
        return _generate_survival(spec, z, rng)
    raise TypeError(f"unsupported outcome specification: {type(spec).__name__}")
