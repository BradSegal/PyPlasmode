"""Distribution checks for weights drawn from the seed."""

import numpy as np
import pytest

from pyplasmode import (
    FeatureSet,
    FeatureSets,
    ModuleTruth,
    Population,
    SampledWeights,
    materialize_truth,
)


@pytest.mark.statistical
def test_sampled_weights_are_uniform_between_the_bounds_with_balanced_signs() -> None:
    names = tuple(f"f{index}" for index in range(2_000))
    population = Population(np.random.default_rng(5).normal(size=(50, len(names))), names)
    structures = FeatureSets((FeatureSet("all", names),))

    truth = materialize_truth(
        ModuleTruth("all", weights=SampledWeights(low=0.5, high=2.5, signs="random")),
        population,
        feature_sets=structures,
        seed=11,
    )
    weights = np.asarray([item.weight for item in truth.weights])
    sizes = np.abs(weights)

    # A uniform draw on [0.5, 2.5] has mean 1.5, variance 1/3 and equal mass in each quarter.
    assert sizes.min() >= 0.5 and sizes.max() <= 2.5
    assert abs(sizes.mean() - 1.5) < 0.05
    assert abs(sizes.var() - 1 / 3) < 0.03
    quarters = np.histogram(sizes, bins=np.linspace(0.5, 2.5, 5))[0] / len(sizes)
    assert np.all(np.abs(quarters - 0.25) < 0.04)
    assert abs((weights > 0).mean() - 0.5) < 0.04
    # Sign and size are drawn independently.
    assert abs(np.corrcoef(np.sign(weights), sizes)[0, 1]) < 0.06
