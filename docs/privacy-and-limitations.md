# Data handling and supported scope

## Empirical data

A plasmode is not "synthetic data" in the privacy sense. Every row of a generated matrix is a
copy of a real row from the source data, and the row indices returned with it map each copy to
its source record. Handle generated matrices, outcomes and row indices under the same access and
sharing conditions as the source dataset. The library returns them in memory and writes nothing
to disk.

## Missing values

The returned feature matrix keeps the missing values of the source rows, so your preprocessing
must handle them as it would on the real data. The built-in signals fill missing values only
inside the calculation of the generating score, with each generating feature's median
([Signal construction](methods.md#signal-construction)); a
[custom function](truths.md#custom-signal) receives the matrix unfilled. A column that is
constant or entirely missing cannot be a generating feature of a built-in signal, because it
cannot carry a signal. It stays in the
matrix, where a model can still select it.

## Data and outcome scope

`Population` accepts a numeric matrix, measured or synthetic, with one sampling unit, such as a
person, per row and one named feature per column, so encode non-numeric variables as numbers
first. Built-in signals standardise each generating feature and treat it as a quantity, so the
encoding decides what the signal means: a one-hot column makes one category, compared with the
rest, a generating feature, and an integer code for an unordered category imposes an order that
does not exist.

The built-in outcomes are binary, continuous, count and ordered categorical responses, as well
as a single time-to-event outcome with outcome-independent censoring. Competing risks, recurrent
events, repeated measurements over time, outcome-dependent censoring and causal interventions
require a custom outcome generator. Compute the generating score with `materialize_truth` and
draw the outcome from it in your own code. The recovery evaluators still apply because they only
require the selections and recovery targets returned from `materialize_truth`.
`evaluate_prediction` still requires a single outcome value per row.
