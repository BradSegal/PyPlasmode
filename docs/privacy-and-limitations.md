# Data handling and supported scope

## Empirical data

A plasmode is not synthetic data in the privacy sense. Every row of a generated matrix is a copy
of a real source row, and the row indices returned with it link each copy to its source record.
Handle generated matrices, outcomes and row indices under the same access and sharing conditions
as the source dataset. The library returns them in memory and writes nothing to disk, so where
they are stored is your decision.

## Missing values

The returned feature matrix keeps the missing values of the source rows, so your preprocessing
must handle them as it would on the real data. PyPlasmode fills missing values only inside the
calculation of the generating score, with each generating feature's median
([Signal construction](methods.md#signal-construction)). A column that is constant or entirely
missing cannot be a generating feature, because it cannot carry a signal. It stays in the
matrix, where a model can still select it.

## Data and outcome scope

`Population` accepts a numeric matrix, measured or synthetic, with one sampling unit, such as a
person, per row and one named feature per column. Encode non-numeric variables as numbers
first. Built-in signals standardise each generating feature and treat it as a quantity, so the
encoding decides what the signal means. A one-hot column makes one category, compared with the
rest, a generating feature. An integer code for an unordered category imposes an order that
does not exist.

The built-in outcomes are binary, continuous, count and ordered-category responses, and a single
event time with censoring that does not depend on the event. Competing risks, recurrent events,
repeated measurements over time, censoring that depends on the outcome and causal interventions
need an outcome generator of your own. Compute the generating score with `materialize_truth`
and draw the outcome from it in your own code. The evaluators still apply, because they need
only the selections and the recovery targets that `materialize_truth` returns.
