# Data handling and supported scope

## Empirical data

Resampling can reproduce individual source records. Apply the source dataset's access and
sharing conditions to generated matrices, outcomes and row indices. The library returns
these arrays in memory; you choose how to store them. Public examples use synthetic inputs.

## Missing values

The returned feature matrix retains its missing values. Median imputation is used internally
to construct the generating signal, so your modelling workflow can apply its own preprocessing.
Signal features need observed variation; constant or entirely missing columns remain in `X`
but cannot generate a non-null signal.

## Simulation scope

PyPlasmode evaluates procedures under a chosen feature distribution, generating signal and
outcome model. Use these choices to represent the application you want to study. The built-in
outcomes cover binary, continuous, count and ordinal responses, plus single-event survival
with independent censoring. Competing risks, recurrent events, longitudinal outcomes,
informative censoring and causal interventions are not implemented in version 0.1.
