# Feature structures and extensions

Some signals and recovery targets refer to features that belong together: a set of correlated
measurements, a biological pathway, a region of an interaction network. Which features belong
together is knowledge about your domain, not something PyPlasmode can infer from the data, so
you supply it from your analysis or an external resource:

- `FeatureSets` holds named groups of features. Correlated and distributed signals are built
  from them ([Generating signals](truths.md)), and `evaluate_groups` scores selections against
  the groups a correlated signal records from them.
- `FeatureGraph` holds features linked by undirected edges, such as a similarity or interaction
  network. `evaluate_graph_region` scores how much of a region of the graph a selection covers
  ([Graph regions](evaluation.md#graph-regions)). An edge can carry a weight for the strength
  of the relationship; the evaluator reports coverage with and without it.

An annotation resource often names features differently from your matrix, or lists features you
did not measure. A signal built from a group with an unmeasured member raises `ValueError`, so
check the match first with `align(population)`. Suppose a pathway lists A, B and G, but your
dataset contains A, B and C:

```python
import numpy as np
import pyplasmode as ppm

population = ppm.Population(np.random.default_rng(7).normal(size=(100, 3)), ("A", "B", "C"))
pathways = ppm.FeatureSets((ppm.FeatureSet("example_pathway", ("A", "B", "G")),))
alignment = pathways.align(population)
print(alignment.mapped, alignment.unmapped, alignment.excluded)
# ('A', 'B') ('G',) ('C',)

members = pathways.get("example_pathway").members
measured = ppm.FeatureSet("example_pathway", tuple(m for m in members if m in alignment.mapped))
print(measured.members)  # ('A', 'B')

aligned_signal = ppm.materialize_truth(
    ppm.ModuleTruth(set_name=measured.name),
    population,
    feature_sets=ppm.FeatureSets((measured,)),
    seed=7,
)
print(tuple(weight.feature for weight in aligned_signal.weights))  # ('A', 'B')
```

`align` sorts names into three lists: `mapped`, in both the annotation and the matrix;
`unmapped`, annotated but not measured; `excluded`, measured but outside the annotation. The next
lines keep only the measured members of the pathway and build a distributed signal from them,
whose generating features are A and B. The alternative is to add a measurement of G.
`align` matches names exactly, so use one identifier convention in both the matrix and the
annotation; otherwise a measured feature appears as unmapped under its other name. A
`FeatureGraph` has the same `align` method.

For a signal structure not covered by the built-in classes, write a function that computes the
generating score and declares its recovery targets ([Custom signal](truths.md#custom-signal)).
