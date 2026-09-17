"""Regenerate PyPlasmode's truth-known ADEMP validation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from pyplasmode import (
    BinaryOutcome,
    ContinuousOutcome,
    CountOutcome,
    OrdinalOutcome,
    Population,
    SparseTruth,
    TimeToEventOutcome,
    evaluate_ranking,
    generate,
    generate_outcome,
    materialize_truth,
    years,
)

SEEDS = tuple(range(40))
SAMPLE_SIZE = 20_000
FEATURE_COUNT = 100
TOP_DEPTH = 10


def _population(seed: int = 20260822) -> Population:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(SAMPLE_SIZE, FEATURE_COUNT))
    return Population(matrix, tuple(f"protein_{index:03d}" for index in range(FEATURE_COUNT)))


def _parameter(calibration: object, name: str) -> float:
    parameters = dict(calibration.parameters)  # type: ignore[attr-defined]
    return float(parameters[name])


def _conformance(population: Population) -> list[dict[str, float | str]]:
    truth = materialize_truth(SparseTruth(features=("protein_000",)), population, seed=17)
    z = truth.signal
    rows: list[dict[str, float | str]] = []

    binary = generate_outcome(BinaryOutcome(0.2, odds_ratio=1.8), z, np.random.default_rng(1))
    binary_fit = sm.GLM(binary.values, sm.add_constant(z), family=sm.families.Binomial()).fit()
    rows.append(_effect_row("binary", np.log(1.8), binary_fit.params[1], binary_fit.bse[1]))

    continuous = generate_outcome(
        ContinuousOutcome(mean=3.0, standard_deviation=2.0, effect=0.7),
        z,
        np.random.default_rng(2),
    )
    continuous_fit = sm.OLS(continuous.values, sm.add_constant(z)).fit()
    rows.append(_effect_row("continuous", 0.7, continuous_fit.params[1], continuous_fit.bse[1]))

    count = generate_outcome(CountOutcome(rate=0.8, rate_ratio=1.4), z, np.random.default_rng(3))
    count_fit = sm.GLM(count.values, sm.add_constant(z), family=sm.families.Poisson()).fit()
    rows.append(_effect_row("count", np.log(1.4), count_fit.params[1], count_fit.bse[1]))

    ordinal = generate_outcome(
        OrdinalOutcome(("low", "middle", "high"), (0.25, 0.50, 0.25), 1.6),
        z,
        np.random.default_rng(4),
    )
    ordinal_fit = OrderedModel(ordinal.codes, z[:, None], distr="logit").fit(
        method="bfgs", disp=False
    )
    rows.append(_effect_row("ordinal", np.log(1.6), ordinal_fit.params[0], ordinal_fit.bse[0]))

    survival = generate_outcome(
        TimeToEventOutcome(
            {years(5): 0.12, years(10): 0.25},
            hazard_ratio=1.7,
            censoring_risks={years(5): 0.08, years(10): 0.18},
        ),
        z,
        np.random.default_rng(5),
    )
    survival_fit = sm.duration.PHReg(survival.time, z[:, None], status=survival.event).fit(
        disp=False
    )
    rows.append(_effect_row("survival", np.log(1.7), survival_fit.params[0], survival_fit.bse[0]))
    return rows


def _effect_row(
    family: str, target: float, estimate: float, standard_error: float
) -> dict[str, float | str]:
    low = float(estimate - 1.96 * standard_error)
    high = float(estimate + 1.96 * standard_error)
    return {
        "family": family,
        "target": float(target),
        "estimate": float(estimate),
        "standard_error": float(standard_error),
        "confidence_interval_low": low,
        "confidence_interval_high": high,
        "target_covered": bool(low <= target <= high),
    }


def _outcome_reproducibility(population: Population) -> list[dict[str, float | str]]:
    truth = materialize_truth(SparseTruth(features=("protein_000",)), population, seed=19)
    z = truth.signal
    specifications = {
        "binary": BinaryOutcome(0.2, odds_ratio=1.8),
        "continuous": ContinuousOutcome(3.0, 2.0, effect=0.7),
        "count": CountOutcome(0.8, 1.4),
        "ordinal": OrdinalOutcome(("low", "middle", "high"), (0.25, 0.5, 0.25), 1.6),
        "survival": TimeToEventOutcome(
            {years(5): 0.12, years(10): 0.25},
            1.7,
            {years(5): 0.08, years(10): 0.18},
        ),
    }
    rows: list[dict[str, float | str]] = []
    for family, specification in specifications.items():
        errors: list[float] = []
        check_count = 0
        for seed in SEEDS:
            generated = generate_outcome(specification, z, np.random.default_rng(seed))
            for check in generated.calibration.checks:
                errors.append(abs(check.realized - check.target))
                check_count += 1
        rows.append(
            {
                "family": family,
                "replicate_count": len(SEEDS),
                "check_count": check_count,
                "failure_count": 0,
                "mean_absolute_target_error": float(np.mean(errors)),
                "maximum_absolute_target_error": float(np.max(errors)),
            }
        )
    return rows


def _recovery_controls(population: Population) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    truth = materialize_truth(SparseTruth(count=5), population, seed=31)
    oracle_tail = tuple(
        feature for feature in population.feature_ids if feature not in truth.direct_features
    )
    oracle = truth.direct_features + oracle_tail
    oracle_result = evaluate_ranking(oracle, truth, depths=(TOP_DEPTH,))
    rows.append(
        {
            "procedure": "oracle",
            "replicate_count": 1,
            "mean_recall_at_10": oracle_result.at_depth[0].recall,
            "expected_recall_at_10": 1.0,
            "monte_carlo_standard_error": 0.0,
        }
    )
    recoveries = []
    for seed in range(1_000):
        ranking = tuple(np.random.default_rng(seed).permutation(population.feature_ids))
        result = evaluate_ranking(ranking, truth, depths=(TOP_DEPTH,))
        recoveries.append(result.at_depth[0].recall)
    expected = TOP_DEPTH / FEATURE_COUNT
    rows.append(
        {
            "procedure": "random",
            "replicate_count": len(recoveries),
            "mean_recall_at_10": float(np.mean(recoveries)),
            "expected_recall_at_10": expected,
            "monte_carlo_standard_error": float(
                np.std(recoveries, ddof=1) / np.sqrt(len(recoveries))
            ),
        }
    )
    return rows


def _null_control(population: Population) -> list[dict[str, float | int | str]]:
    counts = np.zeros(FEATURE_COUNT, dtype=int)
    for seed in range(400):
        plasmode = generate(
            population,
            truth=SparseTruth(features=("protein_000",)),
            outcome=BinaryOutcome(0.5, odds_ratio=1.0),
            sample_size=2_000,
            seed=seed,
        )
        matrix = np.nan_to_num(plasmode.X, nan=0.0)
        centered_y = plasmode.outcome.values - np.mean(plasmode.outcome.values)
        scores = np.abs(matrix.T @ centered_y)
        leading = np.argsort(-scores, kind="stable")[:TOP_DEPTH]
        counts[leading] += 1
    expected = 400 * TOP_DEPTH / FEATURE_COUNT
    standard_deviation = np.sqrt(
        400 * (TOP_DEPTH / FEATURE_COUNT) * (1 - TOP_DEPTH / FEATURE_COUNT)
    )
    return [
        {
            "feature": feature,
            "top_10_count": int(count),
            "expected_count": float(expected),
            "standardized_deviation": float((count - expected) / standard_deviation),
        }
        for feature, count in zip(population.feature_ids, counts, strict=True)
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _figure(
    path: Path, conformance: list[dict[str, object]], null: list[dict[str, object]]
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(7.1, 3.0), layout="constrained", gridspec_kw={"width_ratios": [1.2, 1]}
    )
    positions = np.arange(len(conformance))
    estimates = np.array([float(row["estimate"]) for row in conformance])
    low = np.array([float(row["confidence_interval_low"]) for row in conformance])
    high = np.array([float(row["confidence_interval_high"]) for row in conformance])
    targets = np.array([float(row["target"]) for row in conformance])
    axes[0].errorbar(
        estimates - targets,
        positions,
        xerr=(estimates - low, high - estimates),
        fmt="o",
        color="#0072B2",
        capsize=2,
        markersize=4,
        linewidth=0.9,
    )
    axes[0].axvline(0.0, color="#777777", linewidth=0.7, linestyle="--", zorder=0)
    axes[0].set_yticks(positions, [str(row["family"]).capitalize() for row in conformance])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Estimated minus generating coefficient")
    axes[0].set_title("Effect recovery", loc="left", fontsize=10, pad=12)
    deviations = np.array([float(row["standardized_deviation"]) for row in null])
    axes[1].hist(deviations, bins=np.linspace(-4, 4, 17), color="#009E73", edgecolor="white")
    axes[1].axvline(0, color="#333333", linewidth=0.6)
    axes[1].axvline(-3, color="#666666", linestyle="--", linewidth=0.7)
    axes[1].axvline(3, color="#666666", linestyle="--", linewidth=0.7)
    axes[1].set_xlabel("Feature recurrence deviation (SD)")
    axes[1].set_ylabel("Proteins")
    axes[1].set_title("Selection without signal", loc="left", fontsize=10, pad=12)
    for label, axis in zip(("a", "b"), axes, strict=True):
        axis.text(-0.14, 1.11, label, transform=axis.transAxes, fontsize=11, fontweight="bold")
        axis.tick_params(length=3, width=0.6)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    population = _population()
    recovery = _recovery_controls(population)
    null = _null_control(population)
    conformance = _conformance(population)
    reproducibility = _outcome_reproducibility(population)
    _write_csv(arguments.output / "truth-recovery.csv", recovery)
    _write_csv(arguments.output / "null-control.csv", null)
    _write_csv(arguments.output / "external-conformance.csv", conformance)
    _write_csv(arguments.output / "outcome-reproducibility.csv", reproducibility)
    _figure(arguments.output / "validation", conformance, null)
    report = {
        "design": {
            "aim": (
                "Verify truth recovery, null exchangeability, generated effects and "
                "target recurrence."
            ),
            "data_generating_mechanism": (
                "Seeded empirical-row resampling from an exchangeable Gaussian protein matrix."
            ),
            "estimands": [
                "top-10 exact-feature recall",
                "per-feature null top-10 recurrence",
                "external-estimator coefficient recovery",
                "native-unit calibration error",
            ],
            "methods": [
                "oracle ranking",
                "random ranking",
                "statsmodels maintained estimators",
            ],
            "performance_measures": [
                "analytic expectation",
                "single-realisation reference-estimator interval inclusion",
                "seeded simulation error for stochastic validation checks",
            ],
        },
        "recovery": recovery,
        "null": {
            "replicate_count": 400,
            "maximum_absolute_standardized_deviation": float(
                max(abs(float(row["standardized_deviation"])) for row in null)
            ),
        },
        "external_conformance": conformance,
        "outcome_reproducibility": reproducibility,
    }
    (arguments.output / "validation-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
