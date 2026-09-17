# Feature structures and extensions

`FeatureSets` holds named groups of features, and `FeatureGraph` connects features through
weighted, undirected edges. Groups can represent correlated measurements or biological
pathways; a graph can represent a similarity or interaction network. Supply the memberships
and edges from your own analysis or an external resource.

Call `align(population)` to inspect which feature identities match the population and which
are missing from either side before using the structure for generation or evaluation.

As a biological example, suppose a pathway lists A, B and G, but your dataset contains
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
