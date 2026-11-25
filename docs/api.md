# API reference

Once you know which function you need, you can look up its arguments, defaults and errors here.
The guides explain when to use each: [Methods](methods.md) for generation and
[Evaluation](evaluation.md) for scoring.

Import every function, and every class you construct, from `pyplasmode`; the functions in
`pyplasmode.metrics` are the exception. A few conventions hold throughout:

- **Matrices** have one row per observation and one column per feature. `feature_ids` gives one
  unique name per column, because selections and recovery targets refer to features by name.
- **Generated arrays are read-only**, so that a dataset cannot change after its calibration
  checks have run. Copy an array with `np.array(...)` before changing it.
- **Results are dataclasses.** `dataclasses.asdict` turns one into a plain dictionary, for
  example to save it as a row of a results table.

To keep source rows separate between training, validation and test samples, call
`partition_population` then `generate_partitioned`. The outcome values are in `outcome.values`
for binary, continuous and count outcomes, `outcome.codes` for ordered categories, and
`outcome.time` and `outcome.event` for time to event. Times are in days. The latent event time
(`latent_event_time`) and independent censoring time (`censoring_time`) are drawn before
applying the end of follow-up. Either is infinite if the corresponding event or censoring does
not occur by the last risk time.

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

When features tie at the cutoff of a selection, these functions split the remaining places among
them. They also compute the recovery expected by chance, from random selections that keep the
observed selection probabilities within each matching group ([Matched chance
recovery](evaluation.md#matched-chance-recovery)).

::: pyplasmode.nominations

## Evaluation results

::: pyplasmode.results

## Selection agreement and censored discrimination

::: pyplasmode.metrics
