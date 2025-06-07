"""Type aliases and normalization utilities for ranking types."""
from __future__ import annotations
from typing import Literal, Tuple, Mapping, cast
from enum import Enum

import torch

# Tensor type aliases
FloatTensor = torch.Tensor
LongTensor = torch.Tensor
BoolTensor = torch.Tensor

# Rank types
class RankType(str, Enum):
    """Rank type enumeration."""
    RANK_OPTIMISTIC = "optimistic"
    RANK_PESSIMISTIC = "pessimistic" 
    RANK_REALISTIC = "realistic"

# Ranker types
class RankerType(str, Enum):
    """Ranker type enumeration."""
    COOCCURRENCE = "cooccurrence"
    INDEGREE = "indegree"
    OUTDEGREE = "outdegree"
    INDEGREE_RELATION = "indegree-relation"
    INDEGREE_RELATION_WITH_RELATION = "indegree-relation-with-relation"
    RELAXED_ANCHOR = "relaxed-anchor"
    COMBINED = "combined"
    OUTDEGREE_RELATION = "outdegree-relation"
    PRELAXED_ANCHOR = "p-relaxed-anchor"
    CQD = "cqd"
    RELATION_CO = "relation-co"
    HYBRID = "hybrid"
# Rank type constants
RANK_OPTIMISTIC = RankType.RANK_OPTIMISTIC
RANK_PESSIMISTIC = RankType.RANK_PESSIMISTIC
RANK_REALISTIC = RankType.RANK_REALISTIC

# Ranker type constants
INDEGREE = RankerType.INDEGREE
OUTDEGREE = RankerType.OUTDEGREE
INDEGREE_RELATION = RankerType.INDEGREE_RELATION
RELAXED_ANCHOR = RankerType.RELAXED_ANCHOR
COMBINED = RankerType.COMBINED
CQD = RankerType.CQD
HYBRID = RankerType.HYBRID
RELATION_CO = RankerType.RELATION_CO
# Tuples of valid types
RANK_TYPES = tuple(RankType)
RANKER_TYPES = tuple(RankerType)

# Mapping of alternative names to canonical types
RANK_TYPE_SYNONYMS: dict[str, RankType] = {
    "best": RANK_OPTIMISTIC,
    "worst": RANK_PESSIMISTIC,
    "average": RANK_REALISTIC,
}

RANKER_TYPE_SYNONYMS: dict[str, RankerType] = {
    "in": INDEGREE,
    "out": OUTDEGREE,
    "in_rel": INDEGREE_RELATION,
    "relaxed": RELAXED_ANCHOR,
}

def normalize_rank_type(rank: str | None) -> RankType:
    """Normalize a rank type."""
    if rank is None:
        return RANK_REALISTIC
    rank = rank.lower()
    rank = RANK_TYPE_SYNONYMS.get(rank, rank)
    if rank not in RANK_TYPES:
        raise ValueError(
            f"Invalid target={rank}. Possible values: {RANK_TYPES}")
    return cast(RankType, rank)

def normalize_ranker_type(ranker: str | None) -> RankerType:
    """Normalize a ranker type."""
    if ranker is None:
        return INDEGREE
    ranker = ranker.lower()
    ranker = RANKER_TYPE_SYNONYMS.get(ranker, ranker)
    if ranker not in RANKER_TYPES:
        raise ValueError(
            f"Invalid ranker={ranker}. Possible values: {RANKER_TYPES}")
    return cast(RankerType, ranker)
