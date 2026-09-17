"""Native-unit outcome specification and failure contracts."""

import numpy as np
import pytest

from pyplasmode import (
    BinaryOutcome,
    CalibrationError,
    ContinuousOutcome,
    CountOutcome,
    OrdinalOutcome,
    OutcomeSpec,
    OutcomeSpecificationError,
    TimeToEventOutcome,
    days,
    generate_outcome,
    years,
)


@pytest.mark.unit
def test_outcome_spec_is_part_of_the_public_api() -> None:
    """Users can type model-independent generation functions from the package root."""
    specification: OutcomeSpec = BinaryOutcome(0.1, odds_ratio=1.5)

    assert isinstance(specification, BinaryOutcome)


@pytest.mark.unit
def test_native_outcome_specs_reject_ambiguous_or_infeasible_inputs() -> None:
    with pytest.raises(OutcomeSpecificationError, match="exactly one"):
        BinaryOutcome(0.1, odds_ratio=1.5, risk_ratio=1.2)
    with pytest.raises(OutcomeSpecificationError, match="variance"):
        ContinuousOutcome(0.0, 1.0, effect=2.0)
    with pytest.raises(OutcomeSpecificationError, match="variance_to_mean"):
        CountOutcome(1.0, 1.5, variance_to_mean=0.8)
    with pytest.raises(OutcomeSpecificationError, match="sum to one"):
        OrdinalOutcome(("low", "high"), (0.2, 0.7), 1.2)
    with pytest.raises(OutcomeSpecificationError, match="nondecreasing"):
        TimeToEventOutcome({years(5): 0.2, years(10): 0.1}, hazard_ratio=1.5)


@pytest.mark.unit
def test_duration_units_and_seeded_generation_are_explicit() -> None:
    assert years(1).days == pytest.approx(365.25)
    assert days(7).days == 7.0
    signal = np.linspace(-2.0, 2.0, 1_000)
    spec = BinaryOutcome(0.2, odds_ratio=1.5)

    first = generate_outcome(spec, signal, np.random.default_rng(5))
    second = generate_outcome(spec, signal, np.random.default_rng(5))

    np.testing.assert_array_equal(first.values, second.values)
    assert first.calibration.family == "bernoulli_logit"


@pytest.mark.unit
def test_finite_signal_support_rejects_infeasible_identity_risk() -> None:
    signal = np.linspace(-4.0, 4.0, 1_000)
    with pytest.raises(CalibrationError, match="conditional probabilities"):
        generate_outcome(BinaryOutcome(0.5, risk_difference=0.3), signal, np.random.default_rng(1))
