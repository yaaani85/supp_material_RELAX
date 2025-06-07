
# We only need some graph features and we want to test with multiple graph libraries.
# Besides, we would want to use type annotations in Python.
# Hence, we created these wrappers.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Generic, Hashable, Iterable, Iterator, Optional, Tuple, TypeVar

VertexID = TypeVar("VertexID", bound=Hashable)
VertexLabel = TypeVar('VertexLabel')      # Declare type variable
EdgeID = TypeVar("EdgeID", bound=Hashable)
EdgeLabel = TypeVar('EdgeLabel')      # Declare type variable


class Graph(Generic[VertexID, VertexLabel, EdgeID, EdgeLabel], ABC):
    """Class representing a graph.
    Intentionally, there is no edge and vertex class.
    These object oriented abstractions are usually not as memory efficient because they need a backpointer to the original graph.
    Graphs like this do not support removal, which allows for faster implementations.
    """

    @abstractmethod
    def add_vertex(self, label: VertexLabel) -> VertexID:
        """Add a new Vertex to the graph"""

    @abstractmethod
    def has_vertex(self, vertex: VertexID) -> bool:
        """True in case the graph contains the vertex"""

    @abstractmethod
    def get_vertex_label(self, vertex: VertexID) -> VertexLabel:
        """Get the label of this edge"""

    @abstractmethod
    def get_vertices(self) -> Iterator[VertexID]:
        """Get an iterator over the vertices in the graph"""

    @abstractmethod
    def add_edge(self, source: VertexID, target: VertexID, edge_label: EdgeLabel) -> EdgeID:
        """Add a new edge to the graph"""
    # MC: explicitly removed from the interface. Iterate over the vertices and get their edges.
    # @abstractmethod
    # def edges(self)-> Iterable[Edge[VertexLabel, EdgeLabel]]:
    #     """Get an iterator over the edges in this graph"""

    @abstractmethod
    def get_edge_label_and_target(self, edge: EdgeID) -> Tuple[EdgeLabel, VertexID]:
        """Get the label and target of this edge"""

    def get_edge_label(self, edge: EdgeID) -> EdgeLabel:
        return self.get_edge_label_and_target(edge)[0]

    def get_target(self, edge: EdgeID) -> VertexID:
        return self.get_edge_label_and_target(edge)[1]

    @abstractmethod
    def get_outgoing_edges(self, vertex: VertexID) -> Iterable[EdgeID]:
        """Get the outgoing edges from this Vertex"""

    def iterate_outgoing_edges_label_and_target(self, vertex: VertexID) -> Iterable[Tuple[EdgeLabel, VertexID]]:
        return map(lambda edge: self.get_edge_label_and_target(edge), self.get_outgoing_edges(vertex))


class NaiveGraph(Generic[VertexLabel, EdgeLabel], Graph[int, VertexLabel, Tuple[int, int], EdgeLabel]):
    """A naive (and maybe slow) pure python implementation"""

    def __init__(self) -> None:
        # self._vertex_count: VertexID = VertexID(0)
        self._vertex_labels: list[Optional[VertexLabel]] = []
        self._edges: list[list[Tuple[EdgeLabel, int]]] = []

    def add_vertex(self, label: Optional[VertexLabel] = None) -> int:
        self._vertex_labels.append(label)
        self._edges.append([])
        return len(self._vertex_labels) - 1

    def has_vertex(self, vertex: int) -> bool:
        assert vertex >= 0
        return vertex < len(self._vertex_labels)

    def get_vertex_label(self, vertex: int) -> VertexLabel:
        label = self._vertex_labels[vertex]
        assert label
        return label

    def get_vertices(self) -> Iterator[int]:
        return iter(range(0, len(self._vertex_labels)))

    def add_edge(self, source: int, target: int, edge_label: EdgeLabel) -> Tuple[int, int]:
        outgoing_edges: list[Tuple[EdgeLabel, int]] = self._edges[source]
        edge_info: tuple[EdgeLabel, int] = (edge_label, target)
        outgoing_edges.append(edge_info)
        return source, len(outgoing_edges) - 1

    def get_edge_label_and_target(self, edge: Tuple[int, int]) -> Tuple[EdgeLabel, int]:
        source_vertex = edge[0]
        source_vertex_edges = self._edges[source_vertex]
        edge_index = edge[1]
        edge_info: Tuple[EdgeLabel, int] = source_vertex_edges[edge_index]
        return edge_info

    def get_outgoing_edges(self, vertex: int) -> Iterable[Tuple[int, int]]:
        return map(lambda edge_index: (vertex, edge_index), range(0, len(self._edges[vertex])))

    def iterate_outgoing_edges_label_and_target(self, vertex: int) -> Iterable[Tuple[EdgeLabel, int]]:
        return self._edges[vertex]


class HashVertexGraph(Generic[VertexID, EdgeLabel], Graph[VertexID, VertexID, tuple[VertexID, int], EdgeLabel]):
    """A pure python implementation, which unifies the VertexID and vertexLabel.
      VertexIDs that must be hashable, which also means the label cannot be modified. For censecutive ints, use the NaiveGraph with None for the labels"""

    def __init__(self) -> None:
        #        self._vertex_labels: list[Optional[VertexLabel]] = []
        #        self._edges: list[list[Tuple[EdgeLabel, int]]] = []
        self._edge_map: defaultdict[VertexID, list[tuple[EdgeLabel, VertexID]]] = defaultdict(list)

    def add_vertex(self, label: VertexID) -> VertexID:
        assert label not in self._edge_map
        self._edge_map[label]  # doing the lookup creates the entry
        return label

    def has_vertex(self, vertex: VertexID) -> bool:
        return vertex in self._edge_map

    def get_vertex_label(self, vertex: VertexID) -> VertexID:
        return vertex

    def get_vertices(self) -> Iterator[VertexID]:
        return iter(self._edge_map.keys())

    def add_edge(self, source: VertexID, target: VertexID, edge_label: EdgeLabel) -> Tuple[VertexID, int]:
        assert source in self._edge_map and target in self._edge_map
        edges = self._edge_map[source]
        edges.append((edge_label, target))
        return source, len(edges)

    def get_edge_label_and_target(self, edge: Tuple[VertexID, int]) -> Tuple[EdgeLabel, VertexID]:
        source_vertex, edge_index = edge
        edge_info: tuple[EdgeLabel, VertexID] = self._edge_map[source_vertex][edge_index]
        return edge_info

    def get_outgoing_edges(self, vertex: VertexID) -> Iterable[Tuple[VertexID, int]]:
        return map(lambda edge_index: (vertex, edge_index), range(len(self._edge_map[vertex])))

    def iterate_outgoing_edges_label_and_target(self, vertex: VertexID) -> Iterable[Tuple[EdgeLabel, VertexID]]:
        return self._edge_map[vertex]

    # Might become slow, add counter to add_edge
    def get_total_number_of_edges(self) -> int:
        return sum(len(v) for k, v in self._edge_map.items())
