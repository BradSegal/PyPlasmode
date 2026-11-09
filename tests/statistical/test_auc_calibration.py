"""Expected calibration plus independent realised discrimination over a grid."""

import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from pyplasmode import BinaryOutcome, TimeToEventOutcome, generate_outcome, years
from pyplasmode.metrics import cumulative_dynamic_auc

pytestmark = pytest.mark.statistical


@pytest.mark.parametrize("probability", [0.02, 0.2, 0.5])
@pytest.mark.parametrize("target", [0.5, 0.6, 0.75, 0.9])
def test_binary_expected_and_realised_auc_grid(probability, target):
    rng = np.random.default_rng(771)
    signal = rng.normal(size=80_000)
    spec = BinaryOutcome(probability, target_auc=target)
    outcome = generate_outcome(spec, signal, rng)
    assert outcome.calibration.oracle_auc == pytest.approx(target, abs=1e-7)
    params = dict(outcome.calibration.parameters)
    z = (signal - signal.mean()) / signal.std()
    risk = expit(params["intercept"] + params["signal_coefficient"] * z)
    assert roc_auc_score(outcome.values, risk) == pytest.approx(target, abs=0.025)


@pytest.mark.parametrize("probability", [0.005, 0.05, 0.2])
@pytest.mark.parametrize("censoring", [0, 0.05, 0.3])
@pytest.mark.parametrize("target", [0.5, 0.61, 0.72, 0.79])
def test_survival_expected_and_realised_ipcw_auc_grid(probability, censoring, target):
    rng = np.random.default_rng(883)
    signal = rng.normal(size=120_000)
    horizon = years(10)
    # Follow-up extends beyond the evaluation horizon to provide controls.
    spec = TimeToEventOutcome(
        {horizon: probability, years(11): min(1.1 * probability, 0.99)},
        target_auc=target,
        auc_horizon=horizon,
        censoring_risks={horizon: censoring, years(11): min(1.1 * censoring, 0.99)},
    )
    outcome = generate_outcome(spec, signal, rng)
    assert outcome.calibration.oracle_auc == pytest.approx(target, abs=1e-7)
    risk = dict(outcome.calibration.parameters)["signal_coefficient"] * signal
    actual = cumulative_dynamic_auc(
        outcome.time, outcome.event, outcome.time, outcome.event, risk, horizon=horizon.days
    )
    assert actual == pytest.approx(target, abs=0.035)


def test_expected_survival_auc_is_invariant_to_independent_censoring():
    signal = np.random.default_rng(992).normal(size=5000)
    outputs = [
        generate_outcome(
            TimeToEventOutcome({years(10): 0.005}, target_auc=0.72, censoring_risks={years(10): c}),
            signal,
            np.random.default_rng(3),
        )
        for c in [0, 0.05, 0.3]
    ]
    coefficients = [dict(item.calibration.parameters)["hazard_ratio"] for item in outputs]
    np.testing.assert_array_equal(coefficients, [coefficients[0]] * 3)
