# API reference

The common entry points are exported from `pyplasmode`. Matrices have shape
`(participants, features)`; `feature_ids` supplies one unique string per column.
Generated arrays are read-only. Convert result dataclasses to plain records with
`dataclasses.asdict` for your reporting workflow.

Use `generate` for one sample or `partition_population` followed by
`generate_partitioned` for source-disjoint development and evaluation. Access
`outcome.values` for binary, continuous and count outcomes, `outcome.codes` for
ordinal outcomes, and `outcome.time` plus `outcome.event` for survival outcomes.
Survival times are in days; latent event and censoring times beyond administrative
follow-up are represented by infinity.

The rendered reference includes signatures, types and result fields. See the
methods and evaluation guides for the scientific definitions.

## Populations and sampling

::: pyplasmode.data

## Generation

::: pyplasmode.generate

## Truth specifications

::: pyplasmode.truth

## Outcome specifications and results

::: pyplasmode.outcomes

## Feature sets and graphs

::: pyplasmode.structures

## Ranking and prediction evaluation

::: pyplasmode.evaluate

## Fractional nominations and matched references

::: pyplasmode.nominations

## Evaluation results

::: pyplasmode.results
