"""Feature sets, graphs and alignment to measured columns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyplasmode.data import Population


@dataclass(frozen=True, slots=True)
class StructureAlignment:
    """Explicit alignment of structure identities to a population."""

    mapped: tuple[str, ...]
    unmapped: tuple[str, ...]
    excluded: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """One named set of features."""

    name: str
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("feature-set name must be a non-empty normalized string")
        if not self.members or any(not member for member in self.members):
            raise ValueError("a feature set must contain non-empty identities")
        if len(set(self.members)) != len(self.members):
            raise ValueError("feature-set members must be unique")


@dataclass(frozen=True, slots=True)
class FeatureSets:
    """An ordered collection of named feature sets."""

    sets: tuple[FeatureSet, ...]

    def __post_init__(self) -> None:
        if not self.sets:
            raise ValueError("FeatureSets must contain at least one set")
        names = tuple(item.name for item in self.sets)
        if len(set(names)) != len(names):
            raise ValueError("feature-set names must be unique")

    def get(self, name: str) -> FeatureSet:
        """Return one named feature set or fail explicitly."""
        for item in self.sets:
            if item.name == name:
                return item
        raise ValueError(f"unknown feature set: {name}")

    def align(self, population: Population) -> StructureAlignment:
        """Report mapped, unmapped and population-excluded identities."""
        declared = {member for item in self.sets for member in item.members}
        population_ids = set(population.feature_ids)
        return StructureAlignment(
            mapped=tuple(identity for identity in population.feature_ids if identity in declared),
            unmapped=tuple(sorted(declared - population_ids)),
            excluded=tuple(
                identity for identity in population.feature_ids if identity not in declared
            ),
        )


@dataclass(frozen=True, slots=True)
class FeatureEdge:
    """An undirected feature connection with a finite, non-negative weight."""

    source: str
    target: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("edge endpoints must be distinct non-empty identities")
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("edge weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class FeatureGraph:
    """A weighted undirected graph over named features."""

    nodes: tuple[str, ...]
    edges: tuple[FeatureEdge, ...]

    def __post_init__(self) -> None:
        if not self.nodes or any(not node for node in self.nodes):
            raise ValueError("graph nodes must be non-empty identities")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("graph nodes must be unique")
        node_set = set(self.nodes)
        pairs: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source not in node_set or edge.target not in node_set:
                raise ValueError("every edge endpoint must be declared in graph nodes")
            pair = (
                (edge.source, edge.target)
                if edge.source < edge.target
                else (edge.target, edge.source)
            )
            if pair in pairs:
                raise ValueError("undirected graph edges must be unique")
            pairs.add(pair)

    def align(self, population: Population) -> StructureAlignment:
        """Report mapped, unmapped and population-excluded graph nodes."""
        graph_nodes = set(self.nodes)
        population_ids = set(population.feature_ids)
        return StructureAlignment(
            mapped=tuple(
                identity for identity in population.feature_ids if identity in graph_nodes
            ),
            unmapped=tuple(sorted(graph_nodes - population_ids)),
            excluded=tuple(
                identity for identity in population.feature_ids if identity not in graph_nodes
            ),
        )
