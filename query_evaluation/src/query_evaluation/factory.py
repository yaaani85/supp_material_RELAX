"""Factory functions for creating ranker instances."""
from typing import Optional, List
import sys

from query_evaluation.custom_types import RankerType
from query_evaluation.dataset import QEDataset
from query_evaluation.ranker import Ranker, CQDRanker,HybridRanker, InDegreeRanker, RelationCooccurrenceRanker, OutDegreeRanker,OutDegreeGivenRelationRanker, IndegreeGivenRelationRanker, RelaxedAnchorRanker, CombinedRanker 
from query_evaluation.triple_store import TripleStore, TripleStoreException

def get_ranker(
    ranker_type: RankerType | str, 
    dataset: QEDataset, 
    repository_id: str = None,
    primary_ranker: Ranker = None,
    tie_breakers: tuple[str, ...] | list[Ranker] = None,
) -> Ranker:
    """Create a ranker instance based on the specified type.
    
    Args:
        ranker_type: Type of ranker to instantiate (RankerType or str)
        dataset: Dataset instance containing graph data
        repository_id: Optional repository ID for triple store-based rankers
        tie_breakers: Optional tie breaker types (as strings) or ranker instances
        primary_ranker: Optional primary ranker for CombinedRanker
        
    Returns:
        An instance of the specified ranker
    """
    try:
        if isinstance(ranker_type, str):
            ranker_type = RankerType(ranker_type)

        # Create default primary ranker if needed
        if primary_ranker is None and (ranker_type in {RankerType.COMBINED, RankerType.HYBRID}):
            if repository_id is None:
                raise ValueError("repository_id is required for RelaxedAnchor ranker")
            triple_store = TripleStore(repository_id, dataset)
            primary_ranker = RelaxedAnchorRanker(dataset, triple_store)

        # Handle tie breakers if provided as strings
        if tie_breakers and all(isinstance(tb, str) for tb in tie_breakers):
            # Convert string tie_breakers to rankers
            tie_breaker_rankers = [
                get_ranker(RankerType(tb), dataset, repository_id)
                for tb in tie_breakers
            ] if tie_breakers else [IndegreeGivenRelationRanker(dataset)]  # Default tie breaker

            if ranker_type == RankerType.HYBRID:
                return HybridRanker(dataset, primary_ranker, tie_breaker_rankers)
            elif ranker_type == RankerType.COMBINED:
                return CombinedRanker(dataset, primary_ranker, tie_breaker_rankers)

        # Handle specific ranker types
        if ranker_type == RankerType.RELAXED_ANCHOR:
            if repository_id is None:
                raise ValueError("repository_id is required for RelaxedAnchor ranker")
            triple_store = TripleStore(repository_id, dataset)
            return RelaxedAnchorRanker(dataset, triple_store)
            
        if ranker_type == RankerType.HYBRID:
            return HybridRanker(dataset, primary_ranker, tie_breakers or [IndegreeGivenRelationRanker(dataset)])
            
        if ranker_type == RankerType.COMBINED:
            return CombinedRanker(dataset, primary_ranker, tie_breakers or [IndegreeGivenRelationRanker(dataset)])

        rankers = {
            RankerType.INDEGREE: InDegreeRanker,
            RankerType.OUTDEGREE: OutDegreeRanker,
            RankerType.INDEGREE_RELATION: IndegreeGivenRelationRanker,
            RankerType.OUTDEGREE_RELATION: OutDegreeGivenRelationRanker,
            RankerType.CQD: CQDRanker,
            RankerType.HYBRID: HybridRanker,

            RankerType.RELATION_CO: RelationCooccurrenceRanker,
        }
        
        ranker_class = rankers.get(ranker_type)
        if ranker_class is None:
            raise ValueError(f"Unknown ranker type: {ranker_type}")
        
        return ranker_class(dataset)
        
    except TripleStoreException as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

def _init_ranker(
    dataset: QEDataset,
    primary_ranker_type: str,
    tie_breaker_types: tuple[str, ...],
    repository_id: str
) -> List[Ranker]:
    """Initialize primary ranker and optional tie breakers.
    
    Returns:
        List containing either the primary ranker or a CombinedRanker with tie breakers
    """
    # Get triple store first
    triple_store = dataset.get_triple_store(repository_id)
    
    # Get primary ranker with triple store
    primary_ranker = get_ranker(
        RankerType(primary_ranker_type), 
        dataset, 
        triple_store  # Pass triple_store instead of repository_id
    )
    
    # Get tie breakers if specified
    tie_breaker_rankers = [
        get_ranker(RankerType(tb), dataset, triple_store)  # Pass triple_store here too
        for tb in tie_breaker_types
    ]
    
    if tie_breaker_rankers:
        ranker = CombinedRanker(dataset, primary_ranker, tie_breaker_rankers)
        logging.info(
            f"Initialized CombinedRanker with {primary_ranker.__class__.__name__} "
            f"and tie breakers: {[r.__class__.__name__ for r in tie_breaker_rankers]}"
        )
        return [ranker]
    else:
        logging.info(f"Initialized ranker: {primary_ranker.__class__.__name__}")
        return [primary_ranker]