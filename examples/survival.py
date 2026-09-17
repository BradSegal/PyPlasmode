"""Generate incident outcomes and evaluate an independently fitted Cox procedure.

Requires the validation extra for statsmodels. Model fitting and censoring-aware
prediction metrics belong to the user, not to the simulation library.
"""

import json
from dataclasses import asdict

import numpy as np
from statsmodels.duration.hazard_regression import PHReg

import pyplasmode as ppm


def main() -> None:
    population = ppm.Population(
        np.random.default_rng(1).normal(size=(4_000, 10)),
        tuple(f"protein_{index}" for index in range(10)),
    )
    partition = ppm.partition_population(
        population, training_fraction=0.6, validation_fraction=0.2, seed=7
    )
    samples = ppm.generate_partitioned(
        partition,
        truth=ppm.SparseTruth(features=("protein_0", "protein_1")),
        outcome=ppm.TimeToEventOutcome(
            {ppm.years(5): 0.08, ppm.years(10): 0.20},
            hazard_ratio=2.0,
            censoring_risks={ppm.years(5): 0.02, ppm.years(10): 0.05},
        ),
        sample_sizes=ppm.PartitionSampleSizes(2_400, 800, 800),
        sampling_method="without_replacement",
        seed=42,
    )
    training = samples.training
    if not isinstance(training.outcome, ppm.SurvivalGeneratedOutcome):
        raise TypeError("expected time-to-event outcomes")
    model = PHReg(training.outcome.time, training.X, status=training.outcome.event).fit()
    ranking = tuple(training.feature_ids[index] for index in np.argsort(-np.abs(model.params)))
    result = ppm.evaluate_ranking(ranking, training.truth, depths=(2, 5, 10))
    print(json.dumps({"recovery": asdict(result), "time_unit": "days"}, indent=2))


if __name__ == "__main__":
    main()
