"""Resource-agnostic feature structure contracts."""

import pytest

from pyplasmode import FeatureEdge, FeatureGraph, FeatureSet, FeatureSets, Population


@pytest.mark.unit
def test_feature_sets_report_alignment_without_silent_drops() -> None:
    population = Population([[1.0, 2.0, 3.0]], ("a", "b", "c"))
    sets = FeatureSets((FeatureSet("module", ("a", "b", "outside")),))

    alignment = sets.align(population)

    assert alignment.mapped == ("a", "b")
    assert alignment.unmapped == ("outside",)
    assert alignment.excluded == ("c",)


@pytest.mark.unit
def test_graph_rejects_unknown_endpoints_and_invalid_weights() -> None:
    graph = FeatureGraph(
        nodes=("a", "b", "c"),
        edges=(FeatureEdge("a", "b", 0.8), FeatureEdge("b", "c", 1.0)),
    )
    population = Population([[1.0, 2.0]], ("a", "b"))

    assert graph.align(population).unmapped == ("c",)
    with pytest.raises(ValueError, match="endpoint"):
        FeatureGraph(nodes=("a",), edges=(FeatureEdge("a", "b", 1.0),))
    with pytest.raises(ValueError, match="weight"):
        FeatureEdge("a", "b", -0.1)
