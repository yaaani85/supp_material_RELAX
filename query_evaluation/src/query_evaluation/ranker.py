from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from enum import Enum
import torch
from query_evaluation.dataset import QEDataset
from query_evaluation.graph.graph import HashVertexGraph
from query_evaluation.triple_store import TripleStore
from gqs.loader import QueryGraphBatch
from gqs.query_representation.torch import TorchQuery
from gqs.mapping import RelationMapper, EntityMapper
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import quote
import logging
from query_evaluation.triple_store import TripleStoreException

class Ranker(ABC):
    """Base class for query ranking implementations.
    
    Args:
        dataset (QEDataset): The dataset containing entity and relation mappings
        triple_store (TripleStore, optional): Triple store for query execution
    """
    def __init__(self, dataset: QEDataset, triple_store: Optional[TripleStore] = None) -> None:
        self.dataset = dataset
        self.triple_store = triple_store

    @abstractmethod
    def rank(self, query_idx: int, query: TorchQuery, return_intermediate_entities: bool = False) -> torch.Tensor:
        """Rank entities based on the given query.
        
        Args:
            query (TorchQuery): The query to rank entities for
            
        Returns:
            torch.Tensor: Scores for each entity in the dataset
        """
        pass

    def rank_batch(self, queries: List[TorchQuery]) -> torch.Tensor:
        return torch.stack([self.rank(query_idx, query) for query_idx, query in enumerate(queries)])

def convertRDF_to_ID_graph(rdf_graph: HashVertexGraph[str, str],
                           e2id: EntityMapper, r2id: RelationMapper) -> HashVertexGraph[int, int]:

    converted: HashVertexGraph[int, int] = HashVertexGraph()

    for s in rdf_graph.get_vertices():
        converted.add_vertex(e2id.lookup(s))

    for s in rdf_graph.get_vertices():
        for p, o in rdf_graph.iterate_outgoing_edges_label_and_target(s):
            s_id = e2id.lookup(s)
            p_id = r2id.lookup(p)
            o_id = e2id.lookup(o)
            converted.add_edge(s_id, o_id, p_id)
    return converted


class BaseFrequencyRanker(Ranker, ABC):
    """Abstract base class for frequency-based rankers to reduce code duplication"""
    def __init__(self, dataset: QEDataset, triple_store: TripleStore = None) -> None:
        super().__init__(dataset, triple_store)
        self.graph: HashVertexGraph[int, int] = self._convert_graph(dataset)
        self._frequencies = self._calculate_frequencies()
    
    def _convert_graph(self, dataset: QEDataset) -> HashVertexGraph[int, int]:
        return convertRDF_to_ID_graph(
            dataset.train_and_validation_graph,
            dataset.entity_mapper,
            dataset.relation_mapper
        )
    
    @abstractmethod
    def _calculate_frequencies(self) -> torch.Tensor:
        pass

class InDegreeRanker(BaseFrequencyRanker):
    def _calculate_frequencies(self) -> torch.Tensor:
        frequencies = torch.zeros((self.dataset.entity_mapper.number_of_real_entities()))
        for s in self.graph.get_vertices():
            for p, o in self.graph.iterate_outgoing_edges_label_and_target(s):
                # Only count if relation ID is even (forward relation)
                if p % 2 == 0:
                    frequencies[o] += 1
        # Normalize by number of forward edges only
       
        return frequencies / torch.sum(frequencies)

    def rank(self, query_idx: int, query: TorchQuery) -> torch.Tensor:
        return self._frequencies

class OutDegreeRanker(BaseFrequencyRanker):
    def _calculate_frequencies(self) -> torch.Tensor:
        frequencies = torch.zeros((self.dataset.entity_mapper.number_of_real_entities()))
        for s in self.graph.get_vertices():
            for p, _ in self.graph.iterate_outgoing_edges_label_and_target(s):
                # Only count if relation ID is even (forward relation)
                if p % 2 == 0:
                    frequencies[s] += 1
        # Normalize by number of forward edges only
        forward_edge_count = sum(1 for s in self.graph.get_vertices() 
                               for p, _ in self.graph.iterate_outgoing_edges_label_and_target(s)
                               if p % 2 == 0)
        
        return frequencies / forward_edge_count if forward_edge_count > 0 else frequencies

    def rank(self, query_idx: int, query: TorchQuery) -> torch.Tensor:
        return self._frequencies

class IndegreeGivenRelationRanker(Ranker):
    def __init__(self, dataset: QEDataset, triple_store: TripleStore = None) -> None:
        super().__init__(dataset, triple_store)
        self.num_entities = dataset.entity_mapper.number_of_real_entities()
        self.num_relations = dataset.relation_mapper.number_of_relation_types()

        RDF_graph: HashVertexGraph[str, str] = dataset.train_and_validation_graph
        graph: HashVertexGraph[int, int] = convertRDF_to_ID_graph(
            RDF_graph, dataset.entity_mapper, dataset.relation_mapper)

        vertices = list(graph.get_vertices())
        
        # Initialize frequencies
        self.frequencies_given_relation = torch.zeros((self.num_relations, self.num_entities))
        
        # Calculate frequencies - count all relations
        for s in vertices:
            for p, o in graph.iterate_outgoing_edges_label_and_target(s):
                self.frequencies_given_relation[p][o] += 1

        # Normalize each relation's frequencies
        for r in range(self.num_relations):
            relation_sum = self.frequencies_given_relation[r].sum()
            if relation_sum > 0:
                self.frequencies_given_relation[r] = self.frequencies_given_relation[r] / relation_sum

    def rank(self, query_idx: int, query: TorchQuery) -> torch.Tensor:
        # Find the edge that points to the target
        for edge_id, tail in enumerate(query.edge_index[1]):
            if self.dataset.entity_mapper.is_entity_target(int(tail.item())):
                # Get the relation type for this edge
                edge_type = int(query.edge_type[edge_id].item())
                # Return frequencies for this relation
                return self.frequencies_given_relation[edge_type]
                
  

class OutDegreeGivenRelationRanker(Ranker):
    def __init__(self, dataset: QEDataset, triple_store: TripleStore = None) -> None:
        super().__init__(dataset, triple_store)
        self.num_entities = dataset.entity_mapper.number_of_real_entities()
        self.num_relations = dataset.relation_mapper.number_of_relation_types()

        RDF_graph: HashVertexGraph[str, str] = dataset.train_and_validation_graph
        graph: HashVertexGraph[int, int] = convertRDF_to_ID_graph(
            RDF_graph, dataset.entity_mapper, dataset.relation_mapper)

        vertices = list(graph.get_vertices())
        
        # Initialize frequencies
        self.frequencies_given_relation = torch.zeros((self.num_relations, self.num_entities))
        
        # Calculate frequencies - count outgoing edges for each entity per relation
        for s in vertices:
            for p, o in graph.iterate_outgoing_edges_label_and_target(s):
                self.frequencies_given_relation[p][s] += 1  # Note: counting s not o

        # Normalize each relation's frequencies
        for r in range(self.num_relations):
            relation_sum = self.frequencies_given_relation[r].sum()
            if relation_sum > 0:
                self.frequencies_given_relation[r] = self.frequencies_given_relation[r] / relation_sum

    def rank(self, query_idx: int, query: TorchQuery) -> torch.Tensor:
        # Find the edge that points to the target
        for edge_id, tail in enumerate(query.edge_index[1]):
            if self.dataset.entity_mapper.is_entity_target(int(tail.item())):
                # Get the relation type for this edge
                edge_type = int(query.edge_type[edge_id].item())
                # Return frequencies for this relation
                return self.frequencies_given_relation[edge_type]
                
        # If no target found (shouldn't happen), return uniform distribution
        return torch.ones(self.num_entities) / self.num_entities

class RelaxedAnchorRanker:
    def __init__(self, dataset: QEDataset, triple_store: TripleStore):
        self.dataset = dataset
        self.triple_store = triple_store

    # Public methods first
    def rank(self, query_idx: int, query: TorchQuery, return_intermediate_entities: bool = False) -> Tuple[torch.Tensor, Dict]:
        sparql_query = self._build_relaxed_sparql_query(query)
        self.current_query = sparql_query
        results = self.triple_store.execute_query(sparql_query)
        
        scores = torch.zeros(self.dataset.entity_mapper.number_of_real_entities())
        intermediate_entities = set()
        target_counts = defaultdict(float)  # Use float to accumulate counts
        
        for binding in results.get("results", {}).get("bindings", []):
            target = binding.get("t", {}).get("value")
            intermediate = binding.get("v1", {}).get("value")
            count = float(binding.get("count", {}).get("value", 0))
            
            if target:
                entity_id = self.dataset.entity_mapper.lookup(target)
                target_counts[entity_id] += count  # Accumulate counts for each target
            
            if intermediate:
                int_id = self.dataset.entity_mapper.lookup(intermediate)
                intermediate_entities.add(int_id)
        
        # Set final scores from accumulated counts
        for entity_id, total_count in target_counts.items():
            scores[entity_id] = total_count

        if return_intermediate_entities:
            return scores, {'intermediate_entities': list(intermediate_entities)}
        return scores

    def _analyze_query(self, query: TorchQuery) -> Optional[dict]:
        """Analyze query structure to determine type and components"""
        edge_index = query.edge_index
        edge_types = query.edge_type
        num_edges = query.get_number_of_triples()
        target_edges = []  # (edge_idx, is_target_subject)
        constants = []     # (edge_idx, node_id, is_subject)
        variables = set()  # variable node IDs
        
        for i in range(num_edges):
            subj = edge_index[0, i].item()
            obj = edge_index[1, i].item()
            
            # Track target positions
            if self.dataset.entity_mapper.is_entity_target(subj):
                target_edges.append((i, True))
            if self.dataset.entity_mapper.is_entity_target(obj):
                target_edges.append((i, False))
                
            # Track constants
            if not self.dataset.entity_mapper.is_entity_variable(subj) and not self.dataset.entity_mapper.is_entity_target(subj):
                constants.append((i, subj, True))
            if not self.dataset.entity_mapper.is_entity_variable(obj) and not self.dataset.entity_mapper.is_entity_target(obj):
                constants.append((i, obj, False))
                
            # Track variables
            if self.dataset.entity_mapper.is_entity_variable(subj):
                variables.add(subj)
            if self.dataset.entity_mapper.is_entity_variable(obj):
                variables.add(obj)
        
        return {
            'num_edges': num_edges,
            'edge_index': edge_index,
            'edge_types': edge_types,
            'target_edges': target_edges,
            'constants': constants,
            'variables': variables,
        }
    
    
    # Private methods after
    def _build_relaxed_sparql_query(self, query: TorchQuery) -> Optional[str]:
        query_info = self._analyze_query(query)
        # print(f"Query info: {query_info}")
        if query_info is None:
            return None
            
        match query_info['num_edges']:
            case 1:
                return self._handle_one_edge(
                    query_info['target_edges'], 
                    query_info['edge_types']
                )
            case 2:
                return self._handle_two_edges(
                    query_info['target_edges'], 
                    query_info['constants'],
                    query_info['variables'], 
                    query_info['edge_index'], 
                    query_info['edge_types']
                )
            case 3:
                return self._handle_three_edges(
                    query_info['target_edges'], 
                    query_info['constants'],
                    query_info['variables'], 
                    query_info['edge_index'], 
                    query_info['edge_types']
                )
            case _:
                return None

    def _get_relation_uri(self, edge_type):
        """Format relation URI from edge type"""
        return f"<{self.dataset.relation_mapper.inverse_lookup(edge_type.item())}>"

    def _get_sparql_select_count(self, use_distinct: bool = False) -> str:
        """Helper function to generate SPARQL SELECT and COUNT clause.
        
        Args:
            use_distinct (bool): Whether to use COUNT(DISTINCT ?a) or COUNT(*)
            
        Returns:
            str: SPARQL SELECT clause with appropriate COUNT
        """
        count_clause = "COUNT(DISTINCT ?a)" if use_distinct else "COUNT(*)"
        return f"SELECT ?t ({count_clause} as ?count) WHERE {{\n"

    def _get_sparql_group_by(self) -> str:
        """Helper function to generate SPARQL GROUP BY and ORDER BY clause.
        
        Returns:
            str: Standard GROUP BY and ORDER BY clause
        """
        return "}\nGROUP BY ?t\nORDER BY DESC(?count)"

    def _handle_one_edge(self, target_edges, edge_types):
        """Handle 1p queries: ?a -> target"""
        sparql = self._get_sparql_select_count(use_distinct=False)
        rel_uri = self._get_relation_uri(edge_types[0])
        sparql += f"  ?a {rel_uri} ?t .\n"
        sparql += self._get_sparql_group_by()
        return sparql

    def _handle_intersection(self, target_edges, edge_types):
        """Handle n-way intersection queries (2i, 3i) using optimized subqueries
        2i: ?a1 -> target <- ?a2
        3i: ?a1 -> target <- ?a2 <- ?a3
        """
        sparql = "SELECT ?t ?count WHERE {\n"
        
        # Generate subqueries
        for i in range(len(target_edges)):
            rel_uri = self._get_relation_uri(edge_types[i])
            sparql += f"""
    {{
      SELECT ?t (COUNT(*) as ?inner{i+1}) WHERE {{
        ?a{i} {rel_uri} ?t .
      }} GROUP BY(?t)
    }}"""
        
        # Bind multiplication of counts
        sparql += "\n    BIND("
        count_terms = [f"?inner{i+1}" for i in range(len(target_edges))]
        sparql += "*".join(count_terms)
        sparql += " AS ?count)\n"
        
        sparql += "} ORDER BY DESC(?count)"
        
        return sparql
    
    def _handle_two_edges(self, target_edges, constants, variables, edge_index, edge_types):
        """Handle 2p and 2i queries"""
        if len(target_edges) == 2:
            return self._handle_intersection(target_edges, edge_types)
        
        rel1_uri = self._get_relation_uri(edge_types[0])
        rel2_uri = self._get_relation_uri(edge_types[1])
        
        sparql = self._get_sparql_select_count(use_distinct=False)
        sparql += f"  ?a {rel1_uri} ?v1 .\n"
        sparql += f"  ?v1 {rel2_uri} ?t .\n"
        sparql += self._get_sparql_group_by()
        return sparql

    def _handle_two_i_one_p(self, target_edges, constants, variables, edge_index, edge_types):
        """Handle 2i-1p queries: ?a1 -> var -> target"""
        sparql = self._get_sparql_select_count(use_distinct=False)
        
        rel1_uri = self._get_relation_uri(edge_types[0])
        rel2_uri = self._get_relation_uri(edge_types[1])
        rel3_uri = self._get_relation_uri(edge_types[2])
        
        sparql += f"  ?a1 {rel1_uri} ?v .\n"
        sparql += f"  ?a2 {rel2_uri} ?v .\n"
        sparql += f"  ?v {rel3_uri} ?t .\n"
        
        sparql += self._get_sparql_group_by()
        return sparql

    def _handle_one_p_two_i(self, target_edges, edge_types):
        """Handle 1p-2i queries: ?a1 -> ?v -> target <- ?a2"""
        sparql = self._get_sparql_select_count(use_distinct=False)
        
        rel1_uri = self._get_relation_uri(edge_types[0])
        rel2_uri = self._get_relation_uri(edge_types[1])
        rel3_uri = self._get_relation_uri(edge_types[2])
        
        sparql += f"  ?a1 {rel1_uri} ?v .\n"
        sparql += f"  ?v {rel2_uri} ?t .\n"
        sparql += f"  ?a2 {rel3_uri} ?t .\n"
        
        sparql += self._get_sparql_group_by()
        return sparql

    def _handle_three_edges(self, target_edges, constants, variables, edge_index, edge_types):
        """Handle 3p, 3i, 2i1p, and 1p2i queries"""
        if len(target_edges) == 3:  # 3i case
            return self._handle_intersection(target_edges, edge_types)
        elif len(target_edges) == 2:  # 1p2i case
            # Check if both targets are the same node
            target_positions = set(edge_index[1, pos].item() for pos, _ in target_edges)
            if len(target_positions) == 1:  # Both targets point to same node
                return self._handle_one_p_two_i(target_edges, edge_types)
        elif len(target_edges) == 1:  # Could be 3p or 2i1p
            target_pos = target_edges[0][0]
            
            # Check if it's a 2i1p pattern by looking for two edges pointing to the same variable
            var_counts = defaultdict(int)
            for i in range(edge_index.shape[1]):
                obj = edge_index[1, i].item()
                if obj in variables:
                    var_counts[obj] += 1
            
            # If any variable has 2 incoming edges, it's a 2i1p pattern
            if any(count >= 2 for count in var_counts.values()):
                return self._handle_two_i_one_p(target_edges, constants, variables, edge_index, edge_types)
            
            # Otherwise it's a 3p pattern
            if target_pos == 2:
                sparql = self._get_sparql_select_count(use_distinct=False)
                rel1_uri = self._get_relation_uri(edge_types[0])
                rel2_uri = self._get_relation_uri(edge_types[1])
                rel3_uri = self._get_relation_uri(edge_types[2])
                
                # For 3p, we only relax the anchor
                sparql += f"  ?a {rel1_uri} ?v1 .\n"
                sparql += f"  ?v1 {rel2_uri} ?v2 .\n"
                sparql += f"  ?v2 {rel3_uri} ?t .\n"
                
                sparql += self._get_sparql_group_by()
                return sparql

    def _get_entity_uri(self, entity_id: int) -> str:
        """Convert entity ID to its URI representation for SPARQL queries.
        
        Args:
            entity_id: The internal ID of the entity
            
        Returns:
            Thge full URI string wrapped in < > for SPARQL
        """
        # Get entity URI from mapper
        entity_uri = self.dataset.entity_mapper.inverse_lookup(entity_id)
        
        # Wrap in <> for SPARQL
        return f"<{entity_uri}>"

    def _get_constants(self, query: TorchQuery) -> list[int]:
        """Get all non-target entity IDs from the query."""
        constants = []
        for node_idx in range(query.edge_index.shape[1]):
            # Check both subject and object positions
            subj = query.edge_index[0, node_idx].item()
            obj = query.edge_index[1, node_idx].item()
            
            # Add if not a targete
            if not self.dataset.entity_mapper.is_entity_target(subj):
                constants.append(subj)
            if not self.dataset.entity_mapper.is_entity_target(obj):
                constants.append(obj)
            
        print(f"Query edges: {query.edge_index.tolist()}")  # Debug
        print(f"Constants found: {constants}")  # Debug
        return constants

class CombinedRanker(Ranker):
    """Combines a primary ranker with optional tie-breaking strategies.
    Tie breakers can only break ties within the same score group from the primary ranker.
    
    Args:
        dataset: The dataset containing entity and relation mappings
        primary_ranker: Main ranker that determines initial ordering
        tie_breakers: Optional list of rankers to successively break ties
    """
    def __init__(self, 
                 dataset: QEDataset, 
                 primary_ranker: Ranker,
                 tie_breakers: List[Ranker] = None) -> None:
        super().__init__(dataset, None)
        self.primary_ranker = primary_ranker
        self.tie_breakers = tie_breakers or []

    def rank(self, query_idx: int, query: TorchQuery) -> torch.Tensor:
        try:
            # Get initial scores and convert to ranks (0 = best)
            scores = self.primary_ranker.rank(query_idx, query)
            # Sort scores but preserve ties
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            ranks = torch.zeros_like(scores)
            current_rank = 0
            prev_score = sorted_scores[0]
            ranks[sorted_indices[0]] = current_rank

            # Assign initial ranks, keeping same rank for equal scores
            for i in range(1, len(sorted_indices)):
                if sorted_scores[i] != prev_score:
                    current_rank = i  # Use position as rank instead of incrementing
                    prev_score = sorted_scores[i]
                ranks[sorted_indices[i]] = current_rank
            
            
            # For each tie breaker
            for tie_breaker_idx, tie_breaker in enumerate(self.tie_breakers):
                #print(f"\nProcessing tie breaker {tie_breaker_idx}")
                # Find groups of tied ranks
                unique_ranks = torch.unique(ranks)
                
                for r in unique_ranks:
                    # Get indices of entities with this rank (the tied group)
                    tied_indices = torch.where(ranks == r)[0]
                    
                    if len(tied_indices) > 1:  # Only process if there are actual ties
                        #print(f"Found tied group at rank {r} with size {len(tied_indices)}")
                        
                        # Get tie breaker scores for just the tied entities
                        tiebreaker_scores = tie_breaker.rank(query_idx, query)
                        group_scores = tiebreaker_scores[tied_indices]
                        
                        # Sort tied entities based on tiebreaker scores
                        sorted_scores, sorted_indices = torch.sort(group_scores, descending=True)
                        
                        # Start rank for this group is the minimum position of tied entities
                        base_rank = r
                        current_rank = base_rank
                        prev_score = sorted_scores[0]
                        ranks[tied_indices[sorted_indices[0]]] = current_rank
                        
                        for i in range(1, len(sorted_indices)):
                            if sorted_scores[i] != prev_score:
                                current_rank = base_rank + i  # Use position within group
                                prev_score = sorted_scores[i]
                            ranks[tied_indices[sorted_indices[i]]] = current_rank
                        
                        #print(f"After breaking ties: rank range {base_rank} to {current_rank}")
                
            
            # Convert final ranks back to scores (higher = better)
            final_scores = len(ranks) - ranks
            return final_scores
            
        except Exception as e:
            logging.error(f"Primary ranker failed: {str(e)}")
            if self.tie_breakers:
                return self.tie_breakers[0].rank(query_idx, query)
            return torch.ones(self.dataset.entity_mapper.number_of_real_entities())


class HybridRanker(Ranker):
    def __init__(self, dataset: QEDataset, primary_ranker: Ranker, tie_breakers: Ranker):
        super().__init__(dataset)
        self.primary_ranker = primary_ranker  # Relaxed Anchor
        self.tie_breakers = tie_breakers[0]   # CQD
        self.total_duplicates = 0
        self.total_filtered_targets = 0
        self.total_hard_targets = 0

    def rank(self, query_idx: int, query: TorchQuery) -> torch.Tensor:
        # Get scores and intermediate entities from Relaxed Anchor
        scores, intermediate_info = self.primary_ranker.rank(query_idx, query, return_intermediate_entities=True)
        # Use CQD with these intermediate entities
        scores = scores / torch.sum(scores + 1e-10)
        #print(f"Intermediate entities: {intermediate_info['intermediate_entities']}")
        if intermediate_info['intermediate_entities']:
            cqd_scores = self.tie_breakers.rank(
                query_idx, 
                query, 
                relaxed_targets=intermediate_info['intermediate_entities']
               # relaxed_targets=None
            )
            self.total_duplicates += len(cqd_scores) - len(set(cqd_scores))
            #anchor_mask = (scores == 0)
            

            return cqd_scores
        print("No intermediates found")
        return scores  # Fallback to Relaxed Anchor scores if no intermediates found

class RelationCooccurrenceRanker(Ranker):
    """Ranks entities based on how often their relations co-occur with the target relation in training"""
    
    def __init__(self, dataset: QEDataset, triple_store: TripleStore = None):
        super().__init__(dataset, triple_store)
        self.num_entities = dataset.entity_mapper.number_of_real_entities()
        self.num_relations = dataset.relation_mapper.number_of_relation_types()
        self.cooccur_stats = {}  # target_relation -> {other_rel -> count}
        
        if triple_store:  # SPARQL approach
            self.use_sparql = True
        else:  # Graph traversal approach
            self.use_sparql = False
            RDF_graph: HashVertexGraph[str, str] = dataset.train_and_validation_graph
            self.graph: HashVertexGraph[int, int] = convertRDF_to_ID_graph(
                RDF_graph, dataset.entity_mapper, dataset.relation_mapper)

    def _calculate_cooccurrence_sparql(self, target_relation: int) -> Dict[int, int]:
        target_rel_uri = f"<{self.dataset.relation_mapper.inverse_lookup(target_relation)}>"
        query = f"""
        SELECT ?other_rel (COUNT(*) as ?cooccur) WHERE {{
            ?person {target_rel_uri} ?x .
            ?person ?other_rel ?y .
            FILTER(?other_rel != {target_rel_uri})
        }}
        GROUP BY ?other_rel 
        ORDER BY DESC(?cooccur)
        """
        results = self.triple_store.execute_query(query)
        
        cooccur_counts = defaultdict(int)
        for binding in results.get("results", {}).get("bindings", []):
            rel_uri = binding.get("other_rel", {}).get("value")
            count = int(binding.get("cooccur", {}).get("value", 0))
            rel_id = self.dataset.relation_mapper.lookup(rel_uri)
            cooccur_counts[rel_id] = count
        return cooccur_counts

    def _calculate_cooccurrence(self, target_relation: int) -> Dict[int, int]:
        if self.use_sparql:
            return self._calculate_cooccurrence_sparql(target_relation)
            
        # Original graph traversal implementation
        cooccur_counts = defaultdict(int)
        entities_with_target = set()
        for s in self.graph.get_vertices():
            for p, _ in self.graph.iterate_outgoing_edges_label_and_target(s):
                if p == target_relation:
                    entities_with_target.add(s)
                    break
                    
        for entity in entities_with_target:
            for rel, _ in self.graph.iterate_outgoing_edges_label_and_target(entity):
                if rel != target_relation:
                    cooccur_counts[rel] += 1
        return cooccur_counts

    def rank(self, query_idx: int, query: TorchQuery) -> torch.Tensor:
        # Find the edge that points to the target
        target_relation = None
        for edge_id, tail in enumerate(query.edge_index[1]):
            if self.dataset.entity_mapper.is_entity_target(int(tail.item())):
                target_relation = int(query.edge_type[edge_id].item())
                break
                
        if target_relation not in self.cooccur_stats:
            self.cooccur_stats[target_relation] = self._calculate_cooccurrence(target_relation)
            
        # Score each entity based on its co-occurrence with target relation
        scores = torch.zeros(self.num_entities)
        for entity in self.graph.get_vertices():
            max_cooccur = 0
            # Find the relation with highest co-occurrence
            for rel, _ in self.graph.iterate_outgoing_edges_label_and_target(entity):
                if rel in self.cooccur_stats[target_relation]:
                    max_cooccur = max(max_cooccur, self.cooccur_stats[target_relation][rel])
            scores[entity] = max_cooccur
                
        return scores
