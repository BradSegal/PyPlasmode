# API reference

This page lists the signatures, types and result fields of the public API. See
[Methods](methods.md) for generation and [Evaluation](evaluation.md) for scoring.

The common entry points are exported from `pyplasmode`. Matrices have shape
`(observations, features)`; `feature_ids` supplies one unique string per column.
Generated arrays are read-only. Convert result dataclasses to plain records with
`dataclasses.asdict`.

Use `generate` for one sample or `partition_population` followed by
`generate_partitioned` to keep source rows separate between training, validation and test. Access
`outcome.values` for binary, continuous and count outcomes, `outcome.codes` for
ordinal outcomes, and `outcome.time` plus `outcome.event` for survival outcomes.
Survival times are in days; latent event and censoring times beyond administrative
follow-up are represented by infinity.

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

The `nominations` module represents selections with membership weights, so tied scores
can share a place in a leading list. It also compares recovery with matched random selection.

::: pyplasmode.nominations

## Evaluation results

::: pyplasmode.results

## Selection agreement and censored discrimination

::: pyplasmode.metrics
