"""Calibrate discrimination and compare fixed fitted scores on synthetic data."""

import json
from dataclasses import asdict

import numpy as np
from sklearn.metrics import roc_auc_score

import pyplasmode as ppm
from pyplasmode.metrics import (
    cumulative_dynamic_auc,
    discrimination_gap,
    paired_bootstrap_difference,
)


def main() -> None:
    """Run a binary example and a censoring-aware survival example."""
    rng = np.random.default_rng(12)
    signal = rng.normal(size=20_000)
    binary = ppm.generate_outcome(
        ppm.BinaryOutcome(probability=0.1, target_auc=0.7), signal, np.random.default_rng(3)
    )
    if not isinstance(binary, ppm.BinaryGeneratedOutcome):
        raise TypeError("expected binary outcome")
    fitted = signal + rng.normal(size=signal.size)
    gap = discrimination_gap(
        signal, fitted, metric=lambda scores: roc_auc_score(binary.values, scores)
    )

    def statistic(indices):
        labels = binary.values[indices]
        if np.unique(labels).size < 2:
            return None
        return (roc_auc_score(labels, fitted[indices]), roc_auc_score(labels, signal[indices]))

    uncertainty = paired_bootstrap_difference(signal.size, statistic, resamples=100, seed=7)
    horizon = ppm.years(10)
    survival = ppm.generate_outcome(
        ppm.TimeToEventOutcome(
            {horizon: 0.05, ppm.years(11): 0.06},
            target_auc=0.72,
            auc_horizon=horizon,
            censoring_risks={horizon: 0.05, ppm.years(11): 0.055},
        ),
        signal,
        np.random.default_rng(3),
    )
    if not isinstance(survival, ppm.SurvivalGeneratedOutcome):
        raise TypeError("expected survival outcome")
    auc = cumulative_dynamic_auc(
        survival.time,
        survival.event,
        survival.time,
        survival.event,
        signal,
        horizon=horizon.days,
    )
    print(
        json.dumps(
            {
                "binary_expected_auc": binary.calibration.oracle_auc,
                "binary_gap": asdict(gap),
                "gap_interval": [
                    uncertainty.confidence_interval_low,
                    uncertainty.confidence_interval_high,
                ],
                "survival_expected_auc": survival.calibration.oracle_auc,
                "survival_realised_auc": auc,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
