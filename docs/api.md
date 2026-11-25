# API reference

This page lists every public class and function with its arguments, defaults and errors, for
looking up a detail once you know which function you need. The guides explain when to use each:
[Methods](methods.md) for generation and [Evaluation](evaluation.md) for scoring.

Import every function, and every class you construct, from `pyplasmode`; the functions in
`pyplasmode.metrics` are the exception. A few conventions hold throughout:

- **Matrices** have one row per observation and one column per feature. `feature_ids` gives one
  unique name per column, because selections and recovery targets refer to features by name.
- **Generated arrays are read-only**, so that a dataset cannot change after its calibration
  checks have run. Copy an array with `np.array(...)` before changing it.
- **Results are dataclasses.** `dataclasses.asdict` turns one into a plain dictionary, for
  example to save it as a row of a results table.

`generate` draws one sample. To keep source rows apart between training, validation and test
samples, call `partition_population` and then `generate_partitioned`. The outcome values are in
`outcome.values` for binary, continuous and count outcomes, in `outcome.codes` for ordered
categories, and in `outcome.time` and `outcome.event` for time to event. Times are in days. A
drawn event or censoring time that falls after the last risk time is infinite.

## Populations and sampling

::: pyplasmode.data

## Generation

::: pyplasmode.generate

## Signal specifications

::: pyplasmode.truth

## Outcome specifications and results

::: pyplasmode.outcomes

## Feature sets and graphs

::: pyplasmode.structures

## Ranking and prediction evaluation

::: pyplasmode.evaluate

## Tied selections and matched references

When features tie at the cutoff of a selection, these functions share the remaining places
among them. They also compute the recovery expected from a matched random selection
([Matched chance recovery](evaluation.md#matched-chance-recovery)).

::: pyplasmode.nominations

## Evaluation results

::: pyplasmode.results

## Selection agreement and censored discrimination

::: pyplasmode.metrics
