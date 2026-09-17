"""Define a deterministic nonlinear mechanism without a registry or model adapter."""

import json
from dataclasses import asdict

import numpy as np
from numpy.typing import NDArray

import pyplasmode as ppm


def saturation(
    X: NDArray[np.float64], feature_ids: tuple[str, ...], rng: np.random.Generator
) -> ppm.CustomSignal:
    """Combine two fixed features through a saturating nonlinear response."""
    del rng
    positions = {feature: index for index, feature in enumerate(feature_ids)}
    signal = np.tanh(X[:, positions["a"]]) + 0.5 * np.tanh(X[:, positions["b"]])
    return ppm.CustomSignal(
        signal=signal, direct_features=("a", "b"), description="Two saturating effects"
    )


def main() -> None:
    population = ppm.Population(np.random.default_rng(1).normal(size=(2_000, 4)), tuple("abcd"))
    sample = ppm.generate(
        population,
        truth=ppm.CustomTruth(saturation),
        outcome=ppm.ContinuousOutcome(mean=10, standard_deviation=2, variance_explained=0.25),
        sample_size=2_000,
        seed=42,
        sampling_method="without_replacement",
    )
    oracle = ppm.evaluate_ranking(tuple("abcd"), sample.truth, depths=(2,))
    assert oracle.at_depth[0].recall == 1.0
    print(json.dumps({"oracle_recovery": asdict(oracle)}, indent=2))


if __name__ == "__main__":
    main()
