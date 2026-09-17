"""Compare exact, weighted and group recovery for the same shortlist."""

import numpy as np

import pyplasmode as ppm


def main() -> None:
    population = ppm.Population(
        np.random.default_rng(7).normal(size=(100, 4)), ("A", "B", "C", "D")
    )
    truth = ppm.materialize_truth(
        ppm.SparseTruth(features=("A", "B"), weights=(9.0, 1.0)), population, seed=3
    )
    ranking = ("A", "C", "B", "D")
    exact = ppm.evaluate_ranking(ranking, truth, depths=(2,)).at_depth[0]
    weighted = ppm.evaluate_module(ranking, truth, depths=(2,)).at_depth[0]
    groups = ppm.FeatureSets(
        (ppm.FeatureSet("first", ("A",)), ppm.FeatureSet("second", ("B", "C")))
    )
    grouped_truth = ppm.materialize_truth(
        ppm.CorrelatedTruth(group_names=("first", "second"), sentinels=("A", "B")),
        population,
        feature_sets=groups,
        seed=3,
    )
    group = ppm.evaluate_groups(ranking, grouped_truth, depths=(2,)).at_depth[0]
    print("Exact recovery:", exact)
    print("Generating-weight coverage:", weighted)
    print("Group recovery:", group)


if __name__ == "__main__":
    main()
