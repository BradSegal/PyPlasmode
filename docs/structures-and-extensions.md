# Feature structures and extensions

`FeatureSets` holds named memberships such as correlation groups, complexes or pathways.
`FeatureGraph` holds nodes and weighted, undirected edges. You can create either from your
own analysis or an external annotation resource.

Call `align(population)` to inspect which feature identities match the population and which
are missing from either side before using the structure for generation or evaluation.

For example, suppose an external pathway lists A, B and G, but the proteomic panel measures
A, B and C. All structure types are imported from the public `pyplasmode` namespace:

```python
import numpy as np
import pyplasmode as ppm

population = ppm.Population(np.random.default_rng(7).normal(size=(100, 3)), ("A", "B", "C"))
pathways = ppm.FeatureSets((ppm.FeatureSet("example_pathway", ("A", "B", "G")),))
alignment = pathways.align(population)
print(alignment.mapped)  # ('A', 'B'): present in both
print(alignment.unmapped)  # ('G',): annotation without a measurement
print(alignment.excluded)  # ('C',): measurement outside the annotation

network = ppm.FeatureGraph(nodes=("A", "B"), edges=(ppm.FeatureEdge("A", "B", weight=0.9),))
print(network.align(population).mapped)  # ('A', 'B')
```

Alignment reports identities; it does not impute the missing measurement G. Restrict a
generating pathway to measured members or supply the missing measurement explicitly.
Use the same identifier convention in the matrix and annotation resource. The
[tutorial](tutorial.md) shows how aligned sets become correlated-group signals.

For other generating mechanisms, define a `CustomTruth` function and its recovery targets.
Keep the function focused on the signal calculation; your surrounding workflow can handle
data preparation and modelling. See [Generating signals](truths.md#custom-signal).
