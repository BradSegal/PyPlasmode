"""Scientific checks for the executable gallery source tables."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def gallery():
    path = Path(__file__).resolve().parents[2] / "examples" / "pattern_gallery.py"
    specification = importlib.util.spec_from_file_location("pattern_gallery", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_signal_surfaces_share_coordinates_and_preserve_their_mechanisms(gallery):
    rows = gallery.signal_patterns()
    grouped = {
        name: [row for row in rows if row["mechanism"] == name]
        for name in ("Null", "Sparse", "Distributed", "Product", "Threshold", "Custom")
    }
    assert all(len(values) == 41**2 for values in grouped.values())
    assert all(row["signal"] == 0 for row in grouped["Null"])
    for name, values in grouped.items():
        signal = np.array([row["signal"] for row in values])
        assert abs(signal.mean()) < 1e-12
        if name != "Null":
            assert signal.std() == pytest.approx(1)
    for row in grouped["Sparse"]:
        assert row["signal"] == pytest.approx(row["x1"], abs=1e-12)
    assert len({row["signal"] for row in grouped["Threshold"]}) == 2
    for row in grouped["Product"]:
        assert (
            np.sign(row["signal"]) == np.sign(row["x1"] * row["x2"]) or abs(row["signal"]) < 1e-12
        )


def test_proxy_example_separates_exact_and_group_recovery(gallery):
    cells, rows = gallery.correlated_recovery()
    assert len(cells) == 36
    first = {(row["panel"], row["order"]): row["recovery"] for row in rows if row["depth"] == 1}
    assert first == {
        ("b", "Generator first"): 1,
        ("b", "Substitute first"): 0,
        ("c", "Generator first"): 1,
        ("c", "Substitute first"): 1,
    }


def test_conditional_outcome_curves_follow_calibrated_parameters(gallery):
    rows, calibration = gallery.outcome_patterns()
    for family, result in calibration.items():
        assert all(check["within_tolerance"] for check in result["checks"])
        selected = [row for row in rows if row["panel"] == family]
        if family in ("binary", "ordinal", "survival"):
            assert all(0 <= row["y"] <= 1 for row in selected)
        if family == "ordinal":
            for point in range(101):
                assert sum(row["y"] for row in selected if row["point"] == point) == pytest.approx(
                    1
                )
        if family == "survival":
            for label in ("-1 SD", "0 SD", "+1 SD"):
                values = [row["y"] for row in selected if row["series"] == label]
                assert values[0] == 1
                assert np.all(np.diff(values) <= 0)
            last = {row["series"]: row["y"] for row in selected if row["x"] == 10}
            assert last["-1 SD"] > last["0 SD"] > last["+1 SD"]
