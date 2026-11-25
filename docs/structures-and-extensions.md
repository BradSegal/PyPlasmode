# Feature structures and extensions

Feature structures tell PyPlasmode which features belong together, so a signal or recovery
target can refer to a group or network region. `FeatureSets` stores named feature groups, and
`FeatureGraph` links features through weighted, undirected edges. Groups can represent
correlated measurements or biological pathways. Graphs can represent similarity or interaction
networks. Provide both from your analysis or an external resource.

Call `align(population)` to inspect which feature identities match the population and which
are missing from either side before using the structure for generation or evaluation.

As a biological example, suppose a pathway lists A, B and G, but your dataset contains
A, B and C:

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

Alignment reports identities without imputing the missing measurement G. Restrict a generating
pathway to measured members or provide the missing measurement explicitly. Use the same
identifier convention in the matrix and annotation resource. The [tutorial](tutorial.md)
illustrates how aligned sets become correlated signals.

For a signal structure that the built-in classes do not cover, define a `CustomTruth`
function and its recovery targets. See [Generating signals](truths.md#custom-signal).
