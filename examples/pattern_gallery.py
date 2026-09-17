"""Generate the synthetic source tables used in the PyPlasmode visual gallery."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.special import expit

import pyplasmode as ppm


def saturating_signal(matrix, feature_ids, rng) -> ppm.CustomSignal:
    """Combine two saturating feature responses."""
    return ppm.CustomSignal(
        np.tanh(matrix[:, 0]) + 0.5 * np.tanh(matrix[:, 1]),
        direct_features=feature_ids,
        description="tanh(x1) + 0.5*tanh(x2)",
    )


def signal_patterns() -> list[dict[str, object]]:
    """Evaluate six library mechanisms on the same standardised feature grid."""
    axis = np.linspace(-2.5, 2.5, 41)
    axis /= np.std(axis)
    first, second = np.meshgrid(axis, axis)
    population = ppm.Population(np.column_stack((first.ravel(), second.ravel())), ("P1", "P2"))
    feature_sets = ppm.FeatureSets((ppm.FeatureSet("pair", population.feature_ids),))
    specifications = (
        ("a", "Null", ppm.NullTruth()),
        ("b", "Sparse", ppm.SparseTruth(features=("P1",))),
        ("c", "Distributed", ppm.ModuleTruth("pair")),
        ("d", "Product", ppm.InteractionTruth(("P1", "P2"))),
        ("e", "Threshold", ppm.InteractionTruth(("P1", "P2"), operation="threshold")),
        ("f", "Custom", ppm.CustomTruth(saturating_signal)),
    )
    rows = []
    for panel, name, specification in specifications:
        truth = ppm.materialize_truth(specification, population, seed=7, feature_sets=feature_sets)
        rows.extend(
            {
                "panel": panel,
                "mechanism": name,
                "point": index,
                "x1": float(x1),
                "x2": float(x2),
                "signal": float(signal),
            }
            for index, ((x1, x2), signal) in enumerate(zip(population.X, truth.signal, strict=True))
        )
    return rows


def correlated_recovery() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compare known generating and substitute-first orders on one correlated group."""
    rng = np.random.default_rng(73)
    matrix = rng.normal(size=(1_000, 6))
    matrix[:, 1] = 0.9 * matrix[:, 0] + np.sqrt(1 - 0.9**2) * matrix[:, 1]
    matrix[:, 2] = 0.9 * matrix[:, 0] + np.sqrt(1 - 0.9**2) * matrix[:, 2]
    ids = tuple(f"P{index}" for index in range(1, 7))
    population = ppm.Population(matrix, ids)
    groups = ppm.FeatureSets((ppm.FeatureSet("group", ids[:3]),))
    truth = ppm.materialize_truth(
        ppm.CorrelatedTruth(("group",), sentinels=("P1",)),
        population,
        seed=11,
        feature_sets=groups,
    )
    correlation = np.corrcoef(population.X, rowvar=False)
    cells = [
        {"panel": "a", "row": row, "column": col, "correlation": float(correlation[i, j])}
        for i, row in enumerate(ids)
        for j, col in enumerate(ids)
    ]
    rankings = {
        "Generator first": ids,
        "Substitute first": ("P2", "P4", "P5", "P6", "P3", "P1"),
    }
    rows = []
    for name, ranking in rankings.items():
        exact = ppm.evaluate_ranking(ranking, truth, depths=tuple(range(1, 7)))
        group = ppm.evaluate_groups(ranking, truth, depths=tuple(range(1, 7)))
        for point, group_point in zip(exact.at_depth, group.at_depth, strict=True):
            rows.append(
                {"panel": "b", "order": name, "depth": point.depth, "recovery": point.recall}
            )
            rows.append(
                {
                    "panel": "c",
                    "order": name,
                    "depth": point.depth,
                    "recovery": group_point.group_recall,
                }
            )
    return cells, rows


def outcome_patterns() -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate conditional outcome functions using parameters calibrated by the library."""
    population = ppm.Population(np.random.default_rng(31).normal(size=(2_000, 2)), ("P1", "P2"))
    specifications = {
        "binary": ppm.BinaryOutcome(0.2, odds_ratio=2),
        "continuous": ppm.ContinuousOutcome(40, 5, effect=2),
        "count": ppm.CountOutcome(3, rate_ratio=1.7, variance_to_mean=2),
        "ordinal": ppm.OrdinalOutcome(("Low", "Middle", "High"), (0.3, 0.4, 0.3), 2),
        "survival": ppm.TimeToEventOutcome(
            {ppm.years(5): 0.2, ppm.years(10): 0.45},
            hazard_ratio=2,
            censoring_risks={ppm.years(5): 0.1, ppm.years(10): 0.25},
        ),
    }
    rows = []
    metadata = {}
    grid = np.linspace(-2.5, 2.5, 101)
    for family, specification in specifications.items():
        sample = ppm.generate(
            population,
            truth=ppm.SparseTruth(features=("P1",)),
            outcome=specification,
            sample_size=2_000,
            sampling_method="without_replacement",
            seed=13,
        )
        calibration = sample.outcome.calibration
        metadata[family] = asdict(calibration)
        parameters = dict(calibration.parameters)
        coefficient = float(parameters["signal_coefficient"])
        if family == "binary":
            curves = {"Mean": expit(float(parameters["intercept"]) + coefficient * grid)}
        elif family == "continuous":
            curves = {"Mean": float(parameters["intercept"]) + coefficient * grid}
        elif family == "count":
            curves = {"Mean": np.exp(float(parameters["intercept"]) + coefficient * grid)}
        elif family == "ordinal":
            thresholds = np.asarray(parameters["thresholds"], dtype=float)
            cumulative = expit(thresholds[:, None] - coefficient * grid)
            probabilities = np.diff(
                np.vstack((np.zeros(grid.size), cumulative, np.ones(grid.size))), axis=0
            )
            curves = dict(zip(("Low", "Middle", "High"), probabilities, strict=True))
        else:
            times = np.linspace(0, 10, 101)
            cumulative = np.interp(
                times, (0, 5, 10), (0, *parameters["baseline_cumulative_hazards"])
            )
            curves = {
                label: np.exp(-cumulative * np.exp(coefficient * signal))
                for label, signal in (("-1 SD", -1), ("0 SD", 0), ("+1 SD", 1))
            }
        for label, values in curves.items():
            x = times if family == "survival" else grid
            rows.extend(
                {
                    "panel": family,
                    "series": label,
                    "point": index,
                    "x": float(position),
                    "y": float(value),
                }
                for index, (position, value) in enumerate(zip(x, values, strict=True))
            )
    return rows, metadata


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one rectangular source table."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    correlation, recovery = correlated_recovery()
    curves, calibrations = outcome_patterns()
    for filename, rows in (
        ("signal-shapes.csv", signal_patterns()),
        ("feature-correlations.csv", correlation),
        ("correlated-recovery.csv", recovery),
        ("outcome-families.csv", curves),
    ):
        write_table(args.output / filename, rows)
    (args.output / "outcome-calibration.json").write_text(
        json.dumps(calibrations, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
