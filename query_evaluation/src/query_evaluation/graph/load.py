import logging
from typing import Iterable, TextIO

from query_evaluation.graph import HashVertexGraph

from rdflib import Graph as RDFGraph, Literal


def loadRDF_ignoring_literals(inputs: Iterable[TextIO]) -> HashVertexGraph[str, str]:
    result: HashVertexGraph[str, str] = HashVertexGraph()
    for input in inputs:
        g = RDFGraph()
        g.parse(input)

        logging.info("parsing done")
        for triple in g:
            rdf_s, rdf_p, rdf_o = triple
            if isinstance(rdf_o, Literal):
                logging.info(f"ignoring {triple}")
                continue
            # We do not really care about the urls, or blank node difference

            s, p, o = str(rdf_s), str(rdf_p), str(rdf_o)
            if not result.has_vertex(s):
                result.add_vertex(s)
            if not result.has_vertex(o):
                result.add_vertex(o)
            # we do not have to take care of duplicates, because rdflib already removes these when parsing
            result.add_edge(s, o, p)
    return result


class _LabelCache:
    def __init__(self) -> None:
        self.cache: dict[str, str] = {}

    def get(self, label: str) -> str:
        cached = self.cache.get(label)
        if not cached:
            self.cache[label] = label
            return label
        return cached


def loadTripleFile_non_robust(inputs: Iterable[TextIO]) -> HashVertexGraph[str, str]:
    """
    Read a graph from an input containing triples. This implementation is NOT robust against mistakes in the input.
    It makes the assumption that each line has a triple, stripping of the dot at the end of the line and splitting on whitespace.
    Lines beginning with # are ignored

    Args:
        input (TextIO): The source

    Returns:
        HashVertexGraph[str, str]: a graph with the triples
    """
    result: HashVertexGraph[str, str] = HashVertexGraph()
    label_cache: _LabelCache = _LabelCache()

    for input in inputs:
        for index, triple_string in enumerate(input):
            triple_string = triple_string.strip()
            if triple_string.startswith("#"):
                # ignore
                continue
            assert triple_string[-1] == "."
            triple_string = triple_string[:-1].strip()

            parts = triple_string.split(maxsplit=3)
            s, p, o = label_cache.get(parts[0]), label_cache.get(parts[1]), label_cache.get(parts[2])
            s, p, o = s.strip("<>"), p.strip("<>"), o.strip("<>")
            if not result.has_vertex(s):
                result.add_vertex(s.strip("<>"))
            if not result.has_vertex(o):
                result.add_vertex(o.strip("<>"))
            result.add_edge(s, o, p.strip("<>"))
            if index % 1000000 == 0:
                logging.info(f"done with {index} triples")
    return result
