"""Public AUROC-target specification and infeasibility contracts."""

import numpy as np
import pytest

from pyplasmode import (
    BinaryOutcome,
    CalibrationError,
    OutcomeSpecificationError,
    TimeToEventOutcome,
    generate_outcome,
    years,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("target", [0.49, 1.0, np.nan, np.inf])
def test_invalid_auc_targets(target):
    with pytest.raises(OutcomeSpecificationError, match="target_auc"):
        BinaryOutcome(0.1, target_auc=target)
    with pytest.raises(OutcomeSpecificationError, match="target_auc"):
        TimeToEventOutcome({years(10): 0.1}, target_auc=target)


def test_effect_and_target_are_exclusive_and_horizon_is_declared():
    with pytest.raises(OutcomeSpecificationError, match="exactly one"):
        BinaryOutcome(0.1, odds_ratio=2, target_auc=0.7)
    with pytest.raises(OutcomeSpecificationError, match="exactly one"):
        TimeToEventOutcome({years(10): 0.1}, 2, target_auc=0.7)
    with pytest.raises(OutcomeSpecificationError, match="horizon"):
        TimeToEventOutcome({years(10): 0.1}, target_auc=0.7, auc_horizon=years(11))
    with pytest.raises(OutcomeSpecificationError, match="positive event"):
        TimeToEventOutcome({years(10): 0}, target_auc=0.7)


@pytest.mark.parametrize("survival", [False, True])
def test_constant_signal_rejects_above_chance_but_accepts_chance(survival):
    for target in [0.5, 0.7]:
        spec = (
            TimeToEventOutcome({years(10): 0.1}, target_auc=target)
            if survival
            else BinaryOutcome(0.1, target_auc=target)
        )
        if target > 0.5:
            with pytest.raises(CalibrationError, match="constant"):
                generate_outcome(spec, np.zeros(100), np.random.default_rng(1))
        else:
            outcome = generate_outcome(spec, np.zeros(100), np.random.default_rng(1))
            assert outcome.calibration.oracle_auc == pytest.approx(0.5)
            assert dict(outcome.calibration.parameters)["signal_coefficient"] == 0


def test_unattainable_tied_signal_target_is_not_clipped():
    signal = np.repeat([0.0, 1.0], 50)
    with pytest.raises(CalibrationError, match="unattainable"):
        generate_outcome(BinaryOutcome(0.1, target_auc=0.999), signal, np.random.default_rng(2))


def test_seed_reproducibility_and_reported_solved_effect():
    signal = np.random.default_rng(9).normal(size=2000)
    spec = BinaryOutcome(0.1, target_auc=0.7)
    a, b = [generate_outcome(spec, signal, np.random.default_rng(2)) for _ in range(2)]
    np.testing.assert_array_equal(a.values, b.values)
    assert a.calibration == b.calibration
    assert a.calibration.oracle_auc == pytest.approx(0.7, abs=1e-8)
    assert a.calibration.target_auc == 0.7
    assert dict(a.calibration.parameters)["odds_ratio"] > 1


def test_target_at_supported_bracket_endpoint_never_exceeds_the_bracket():
    signal = np.r_[np.zeros(99), 1.0]
    limit = 8.0
    fixed = generate_outcome(
        BinaryOutcome(0.1, odds_ratio=np.exp(limit)), signal, np.random.default_rng(2)
    )
    target = fixed.calibration.oracle_auc
    calibrated = generate_outcome(
        BinaryOutcome(0.1, target_auc=target), signal, np.random.default_rng(2)
    )
    assert dict(calibrated.calibration.parameters)["signal_coefficient"] <= limit
    assert calibrated.calibration.oracle_auc == pytest.approx(target, abs=1e-10)


def test_target_just_above_supported_bracket_is_rejected():
    signal = np.linspace(-3, 3, 1000)
    fixed = generate_outcome(
        BinaryOutcome(0.1, odds_ratio=np.exp(8)), signal, np.random.default_rng(2)
    )
    with pytest.raises(CalibrationError, match="supported log-ratio bracket"):
        generate_outcome(
            BinaryOutcome(0.1, target_auc=fixed.calibration.oracle_auc + 5e-11),
            signal,
            np.random.default_rng(2),
        )


@pytest.mark.parametrize("survival", [False, True])
@pytest.mark.parametrize("outlier", [30.0, -30.0, 500.0])
def test_target_bracket_is_not_shrunk_by_one_outlier(survival, outlier):
    signal = np.random.default_rng(44).normal(size=50000)
    signal[0] = outlier
    spec = (
        TimeToEventOutcome({years(10): 0.1}, target_auc=0.78)
        if survival
        else BinaryOutcome(0.1, target_auc=0.78)
    )
    outcome = generate_outcome(spec, signal, np.random.default_rng(2))
    assert outcome.calibration.oracle_auc == pytest.approx(0.78, abs=1e-8)
    assert outcome.calibration.checks[0].expected == pytest.approx(0.1, abs=1e-9)


@pytest.mark.parametrize("target", [None, 0.72])
def test_default_survival_auc_horizon_can_be_evaluated(target):
    from pyplasmode.metrics import cumulative_dynamic_auc

    spec = TimeToEventOutcome(
        {years(10): 0.1}, hazard_ratio=2 if target is None else None, target_auc=target
    )
    assert spec.auc_horizon.days == years(5).days
    signal = np.random.default_rng(7).normal(size=10000)
    outcome = generate_outcome(spec, signal, np.random.default_rng(3))
    auc = cumulative_dynamic_auc(
        outcome.time,
        outcome.event,
        outcome.time,
        outcome.event,
        signal,
        horizon=spec.auc_horizon.days,
    )
    assert 0.5 < auc < 1
    if target is not None:
        assert outcome.calibration.oracle_auc == pytest.approx(target, abs=1e-8)
        assert auc == pytest.approx(target, abs=0.04)


def test_explicit_administrative_horizon_and_interior_hazard_interpolation():
    spec = TimeToEventOutcome({years(10): 0.1}, target_auc=0.72, administrative_horizon=years(11))
    assert spec.auc_horizon == years(10)
    interior = TimeToEventOutcome(
        {years(5): 0.05, years(10): 0.1}, target_auc=0.7, auc_horizon=years(7)
    )
    result = generate_outcome(interior, np.linspace(-3, 3, 10000), np.random.default_rng(3))
    assert result.calibration.oracle_auc == pytest.approx(0.7, abs=1e-8)
    signal = np.linspace(-3, 3, 10000)
    outcome = generate_outcome(spec, signal, np.random.default_rng(3))
    assert outcome.time.max() == years(11).days
    assert outcome.calibration.oracle_auc == pytest.approx(0.72, abs=1e-8)
    for kwargs in ({"auc_horizon": years(10)}, {"administrative_horizon": years(9)}):
        with pytest.raises(OutcomeSpecificationError, match="horizon"):
            TimeToEventOutcome({years(10): 0.1}, hazard_ratio=2, **kwargs)


def test_parameter_accessor_returns_fresh_mapping_and_preserves_immutability():
    result = generate_outcome(
        BinaryOutcome(0.1, odds_ratio=2), np.linspace(-2, 2, 1000), np.random.default_rng(2)
    ).calibration
    assert result["odds_ratio"] == result.parameter_dict["odds_ratio"] == 2
    copy = result.parameter_dict
    copy["odds_ratio"] = 99
    assert result["odds_ratio"] == 2
    with pytest.raises(KeyError):
        result["absent"]


@pytest.mark.parametrize("ratio", [0.5, 2.0])
def test_protective_oracle_score_uses_the_solved_coefficient(ratio):
    from sklearn.metrics import roc_auc_score

    signal = np.random.default_rng(5).normal(size=20000)
    outcome = generate_outcome(
        BinaryOutcome(0.2, odds_ratio=ratio), signal, np.random.default_rng(2)
    )
    beta = float(outcome.calibration["signal_coefficient"])
    assert roc_auc_score(outcome.values, beta * signal) > 0.5
    assert (beta < 0) == (ratio < 1)
