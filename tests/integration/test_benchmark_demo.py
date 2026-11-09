"""Exercise the public CLI and check its scientific output contract."""

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
def test_benchmark_demo(tmp_path: Path) -> None:
    """All mechanisms, effects and models yield calibrated, auditable rows."""
    pytest.importorskip("sksurv")
    pytest.importorskip("xgboost")
    example = Path(__file__).parents[2] / "examples" / "benchmark_recovery_demo.py"
    output = tmp_path / "results.csv"
    result = subprocess.run(
        [sys.executable, str(example), "--rows", "1000", "--output", str(output)],
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    assert "AUROC Oracle Recovery Chance Adjusted Top20" in result.stdout
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 27
    assert {(r["scenario"], r["target_oracle_auc"], r["method"]) for r in rows} == {
        (scenario, str(target), method)
        for scenario in ("sparse", "correlated", "distributed")
        for target in (0.6, 0.7, 0.8)
        for method in ("Cox", "ridge", "XGBoost")
    }
    for row in rows:
        assert float(row["expected_oracle_auc"]) == pytest.approx(
            float(row["target_oracle_auc"]), abs=1e-6
        )
        assert 0 <= float(row["auroc"]) <= 1
        assert 0 <= float(row["realised_oracle_auc"]) <= 1
        assert 0 < int(row["top20_size"]) <= 20
        assert np.isfinite(float(row["adjusted_recovery"]))
        assert float(row["adjusted_recovery"]) == pytest.approx(
            float(row["recovery"]) - float(row["chance_recovery"])
        )
        assert (
            row["recovery_estimand"]
            == {"sparse": "exact", "correlated": "group", "distributed": "module"}[row["scenario"]]
        )
        if row["scenario"] in {"sparse", "distributed"}:
            assert float(row["chance_recovery"]) == pytest.approx(int(row["top20_size"]) / 120)
        if row["scenario"] == "distributed":
            assert 0 <= float(row["recovery"]) <= 20 / 60 + 1e-12
