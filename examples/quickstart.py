"""Run the documented sparse-biomarker quickstart."""

import json
from dataclasses import asdict

import numpy as np
from sklearn.linear_model import LogisticRegression

import pyplasmode as ppm


def main() -> None:
    """Generate, fit and evaluate one fully synthetic example."""
    rng = np.random.default_rng(1)
    population = ppm.Population(
        rng.normal(size=(2_000, 100)), tuple(f"protein_{index}" for index in range(100))
    )
    partition = ppm.partition_population(
        population, training_fraction=0.6, validation_fraction=0.2, seed=7
    )
    samples = ppm.generate_partitioned(
        partition,
        truth=ppm.SparseTruth(features=("protein_3", "protein_17", "protein_42")),
        outcome=ppm.BinaryOutcome(probability=0.15, odds_ratio=2.0),
        sample_sizes=ppm.PartitionSampleSizes(1_200, 400, 400),
        seed=42,
        sampling_method="without_replacement",
    )
    training, evaluation = samples.training, samples.evaluation
    if not isinstance(training.outcome, ppm.BinaryGeneratedOutcome) or not isinstance(
        evaluation.outcome, ppm.BinaryGeneratedOutcome
    ):
        raise RuntimeError("quickstart expected a binary outcome")
    model = LogisticRegression(max_iter=1_000).fit(training.X, training.outcome.values)
    ranking = tuple(training.feature_ids[index] for index in np.argsort(-np.abs(model.coef_[0])))
    recovery = ppm.evaluate_ranking(ranking, training.truth, depths=(10, 20, 50))
    prediction = ppm.evaluate_prediction(
        evaluation.outcome.values, model.predict_proba(evaluation.X)[:, 1], metric="roc_auc"
    )
    print(
        json.dumps(
            {"held_out_prediction": asdict(prediction), "recovery": asdict(recovery)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
