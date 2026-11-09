"""Compare prediction and top-20 recovery using entirely synthetic features.

Install the optional ``benchmark`` extra for Cox and XGBoost. No data downloads,
accelerator, plotting library or private analysis code are needed.
"""

import argparse
import csv
import sys
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import pyplasmode as ppm
from pyplasmode.metrics import cumulative_dynamic_auc


def run_demo(*, rows: int = 3_000, seed: int = 42) -> list[dict[str, object]]:
    """Return all 27 comparisons with fixed settings and a held-out evaluation.

    Oracle calibration uses the finite synthetic covariate population before
    outcome sampling. It targets expected discrimination, not realised test AUROC.
    """
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv
        from xgboost import XGBClassifier
    except ImportError as error:
        raise SystemExit(
            "Install the optional models: pip install 'pyplasmode[benchmark]'"
        ) from error
    if rows < 1_000:
        raise ValueError("use at least 1,000 rows for this demonstration")
    rng = np.random.default_rng(seed)
    feature_ids = tuple(f"feature_{i:03d}" for i in range(120))
    # Shared latent factors give correlation 0.7 within each four-feature block.
    latent = rng.normal(size=(rows, 30))
    X = np.sqrt(0.7) * np.repeat(latent, 4, axis=1)
    X += np.sqrt(0.3) * rng.normal(size=X.shape)
    population = ppm.Population(X, feature_ids)
    groups = tuple(ppm.FeatureSet(f"block_{i}", feature_ids[4 * i : 4 * i + 4]) for i in range(30))
    structures = ppm.FeatureSets((*groups, ppm.FeatureSet("distributed", feature_ids[:60])))
    sentinels = tuple(feature_ids[4 * i] for i in range(8))
    scenarios = (
        ("sparse", ppm.SparseTruth(features=sentinels), "exact"),
        (
            "correlated",
            ppm.CorrelatedTruth(group_names=tuple(g.name for g in groups[:8]), sentinels=sentinels),
            "group",
        ),
        ("distributed", ppm.ModuleTruth(set_name="distributed"), "module"),
    )
    indices = rng.permutation(rows)
    train, test = indices[: int(0.6 * rows)], indices[int(0.6 * rows) :]
    scaler = StandardScaler().fit(X[train])
    training, evaluation = scaler.transform(X[train]), scaler.transform(X[test])
    horizon, end = ppm.years(10), ppm.years(11)
    results = []
    with threadpool_limits(limits=1):
        for scenario, specification, estimand in scenarios:
            truth = ppm.materialize_truth(
                specification, population, feature_sets=structures, seed=seed
            )
            for target in (0.60, 0.70, 0.80):
                outcome = ppm.generate_outcome(
                    ppm.TimeToEventOutcome(
                        {horizon: 0.20, end: 0.22}, target_auc=target, auc_horizon=horizon
                    ),
                    truth.signal,
                    # Common random numbers across effect sizes and scenarios.
                    np.random.default_rng(seed + 1),
                )
                if not isinstance(outcome, ppm.SurvivalGeneratedOutcome):
                    raise TypeError("expected a survival outcome")
                labels = (outcome.time <= horizon.days) & outcome.event
                cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(
                    training, Surv.from_arrays(outcome.event[train], outcome.time[train])
                )
                ridge = Ridge(alpha=10.0).fit(training, labels[train].astype(float))
                boosted = XGBClassifier(
                    n_estimators=80,
                    max_depth=2,
                    learning_rate=0.05,
                    tree_method="hist",
                    n_jobs=1,
                    random_state=seed,
                    importance_type="gain",
                ).fit(training, labels[train])

                auc = partial(
                    cumulative_dynamic_auc,
                    outcome.time[train],
                    outcome.event[train],
                    outcome.time[test],
                    outcome.event[test],
                    horizon=horizon.days,
                )

                oracle_auc = auc(truth.signal[test])
                methods = (
                    ("Cox", cox.predict(evaluation), np.abs(cox.coef_)),
                    ("ridge", ridge.predict(evaluation), np.abs(ridge.coef_)),
                    (
                        "XGBoost",
                        boosted.predict_proba(evaluation)[:, 1],
                        boosted.feature_importances_,
                    ),
                )
                for method, prediction, importance in methods:
                    nomination = ppm.nomination_from_scores(feature_ids, importance, depth=20)
                    recovery = ppm.evaluate_matched_recovery(
                        nomination, truth, estimand=estimand, strata=("all",) * len(feature_ids)
                    )
                    results.append(
                        {
                            "scenario": scenario,
                            "target_oracle_auc": target,
                            "expected_oracle_auc": outcome.calibration.oracle_auc,
                            "realised_oracle_auc": oracle_auc,
                            "method": method,
                            "auroc": auc(prediction),
                            "recovery_estimand": estimand,
                            "top20_size": recovery.achieved_size,
                            "recovery": recovery.observed_recovery,
                            "chance_recovery": recovery.expected_recovery,
                            "adjusted_recovery": recovery.chance_adjusted_recovery,
                        }
                    )
    return results


def main() -> None:
    """Print an AUROC-versus-recovery table and optionally save its full values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="optional CSV destination")
    args = parser.parse_args()
    results = run_demo(rows=args.rows, seed=args.seed)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    print("Synthetic data only; fixed fits, one split, no uncertainty intervals.")
    print("Adjusted recovery = observed recovery minus analytic chance recovery.")
    print("Scenario     Target Method    AUROC Oracle Recovery Chance Adjusted Top20")
    for row in results:
        print(
            f"{row['scenario']:<12} {row['target_oracle_auc']:.2f}   {row['method']:<8} "
            f"{row['auroc']:.3f} {row['realised_oracle_auc']:.3f}  "
            f"{row['recovery']:.3f}    {row['chance_recovery']:.3f}  "
            f"{row['adjusted_recovery']:.3f}    {row['top20_size']:2d}"
        )


if __name__ == "__main__":
    sys.exit(main())
