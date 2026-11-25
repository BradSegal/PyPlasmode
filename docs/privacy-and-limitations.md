# Data handling and supported scope

## Empirical data

Resampling can reproduce individual source records. Apply the source dataset's access and
sharing conditions to generated matrices, outcomes and row indices. The library returns
these arrays in memory; you choose how to store them.

## Missing values

The returned feature matrix retains its missing values, so your modelling workflow can apply
its own preprocessing. For the built-in signal structures, median imputation is used only
internally, to compute each row's generating score from the generating features. Constant or entirely missing columns remain
in `X` but cannot be generating features; see
[Signal construction](methods.md#signal-construction).

## Data and outcome scope

`Population` accepts a numeric matrix with one sampling unit per row and one named feature
per column. It can contain measured or synthetic data. Encode non-numeric variables before
passing them to the library, and choose a signal appropriate to that encoding.

PyPlasmode evaluates procedures under a chosen feature distribution, signal and outcome model.
Choose these to reflect your application. The built-in outcomes include binary, continuous,
count and ordinal responses, plus single-event survival with independent censoring. Competing
risks, recurrent events, longitudinal outcomes, informative censoring and causal interventions
require a different outcome generator.
