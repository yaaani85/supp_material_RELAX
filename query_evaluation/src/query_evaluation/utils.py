from pathlib import Path
from typing import Dict, List, TypeVar
import os
from gqs.mapping import RelationMapper, EntityMapper

import logging
from query_evaluation.graph.graph import HashVertexGraph
from query_evaluation.metrics import Metrics
from query_evaluation.ranker import CombinedRanker
from query_evaluation.factory import get_ranker
from query_evaluation.custom_types import RankerType
from query_evaluation.dataset import QEDataset
from gqs.query_representation.torch import TorchQuery



def _get_output_path_suffix(
    base_path: str,
    ranker: str,
    tie_breakers: tuple[str, ...],
    query_type: str,
    repository_id: str
) -> str:
    """Constructs the output file path with appropriate suffixes.
    
    Example: results_relaxed_anchor_ties_cooccurrence_indegree_1hop_dbpedia.json
    """
    name, ext = os.path.splitext(base_path)
    parts = [name, ranker]
    
    if tie_breakers:
        parts.append('ties_' + '_'.join(tie_breakers))
    if query_type != 'all':
        parts.append(query_type)
    parts.append(repository_id)
    
    return f"{'_'.join(parts)}{ext}"

def save_results_to_file(metrics: Dict[str, Metrics], output_path: str | Path) -> None:
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary mapping ranker names to their metrics
        output_path: Path where to save the results
    """
    # Convert metrics to JSON-serializable format
    serializable_metrics = {
        ranker_name: metric.metrics  # Use the internal metrics dictionary
        for ranker_name, metric in metrics.items()
    }
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    import json
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"Results saved to {output_path}")

T = TypeVar('T')
U = TypeVar('U')

def convert_rdf_to_id_graph(
    rdf_graph: HashVertexGraph[str, str],
    e2id: EntityMapper,
    r2id: RelationMapper
) -> HashVertexGraph[int, int]:
    """Convert a string-based RDF graph to an ID-based graph.
    
    Args:
        rdf_graph: Input graph with string vertices and edges
        e2id: Entity mapper for converting entity strings to IDs
        r2id: Relation mapper for converting relation strings to IDs
        
    Returns:
        Converted graph with integer vertices and edges
        
    Raises:
        ValueError: If RDF graph is None
        RuntimeError: If conversion fails
    """
    if rdf_graph is None:
        raise ValueError("RDF graph cannot be None")
        
    converted: HashVertexGraph[int, int] = HashVertexGraph()
    
    try:
        for s in rdf_graph.get_vertices():
            converted.add_vertex(e2id.lookup(s))

        for s in rdf_graph.get_vertices():
            for p, o in rdf_graph.iterate_outgoing_edges_label_and_target(s):
                s_id = e2id.lookup(s)
                p_id = r2id.lookup(p)
                o_id = e2id.lookup(o)
                converted.add_edge(s_id, o_id, p_id)
        return converted
    except Exception as e:
        raise RuntimeError(f"Failed to convert RDF graph: {str(e)}")

def get_key_for_query(query: TorchQuery) -> None:
    """Store scores based on query type patterns"""
    edge_index = query.edge_index
    edge_type = query.edge_type
    
    # For 1p pattern: (e, (r,))
    if len(edge_type) == 1:
        start_entity = edge_index[0,0].item()
        relations = tuple(r.item() for r in edge_type)
        key = (start_entity, relations)
        
    # For 2p pattern: (e, (r,r))
    elif len(edge_type) == 2 and edge_index[0,0].item() != edge_index[0,1].item():
        start_entity = edge_index[0,0].item()
        relations = tuple(r.item() for r in edge_type)
        key = (start_entity, relations)
        
    # For 3p pattern: (e, (r,r,r))
    elif len(edge_type) == 3 and len(set(edge_index[0].tolist())) == 1:
        start_entity = edge_index[0,0].item()
        relations = tuple(r.item() for r in edge_type)
        key = (start_entity, relations)
        
    # For 2i pattern: ((e, (r,)), (e, (r,)))
    elif len(edge_type) == 2 and edge_index[0,0].item() == edge_index[0,1].item():
        start_entity = edge_index[0,0].item()
        relations = tuple(r.item() for r in edge_type)
        key = (start_entity, relations)
        
    # For 3i pattern: ((e, (r,)), (e, (r,)), (e, (r,)))
    elif len(edge_type) == 3 and len(set(edge_index[0].tolist())) == 3:
        start_entities = tuple(edge_index[0].tolist())
        relations = tuple(r.item() for r in edge_type)
        key = (start_entities, relations)
        
    # For 1p-2i pattern: (('e', ('r', 'r')), ('e', ('r',)))
    elif len(edge_type) == 3 and edge_index[1,1].item() == edge_index[1,2].item():  # Check if last two edges point to same target
        start_entities = (edge_index[0,0].item(), edge_index[0,2].item())
        relations = tuple(r.item() for r in edge_type)
        key = (start_entities, relations)
        
    # For 2i-1p pattern: ((('e', ('r',)), ('e', ('r',))), ('r',))
    elif len(edge_type) == 3 and edge_index[1,0].item() == edge_index[1,1].item():  # Check if first two edges point to same intermediate
        start_entities = (edge_index[0,0].item(), edge_index[0,1].item())
        relations = tuple(r.item() for r in edge_type)
        key = (start_entities, relations)
    

    return key