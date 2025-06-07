"""Evaluation utilities."""
from __future__ import annotations
import dataclasses
import logging
from abc import abstractmethod
from query_evaluation.custom_types import RANK_OPTIMISTIC, RANK_PESSIMISTIC, RANK_REALISTIC, RANK_TYPES
from tqdm.auto import tqdm
from gqs.loader import QueryGraphBatch
from query_evaluation.utils import get_key_for_query
# from .data.mapping import get_entity_mapper
from query_evaluation.custom_types import FloatTensor, LongTensor
import torch
import pandas
from query_evaluation.metrics import Metrics
from typing import Collection, List, Mapping, MutableMapping, Optional, Tuple, Union, cast, Dict
import numpy

__all__ = [
    "RankingMetricAggregator"]

logger = logging.getLogger(__name__)

MICRO_AVERAGE = "micro"
MACRO_AVERAGE = "macro"

@dataclasses.dataclass
class _Ranks:
    """Rank results."""
    #: The optimistic rank (i.e. best among equal scores)
    optimistic: LongTensor

    #: The pessimistic rank (i.e. worst among equal scores)
    pessimistic: LongTensor

    #: The expected rank
    expected_rank: Optional[FloatTensor] = None

    # Complete ranking for analysis
    ranking: Optional[FloatTensor] = None

    # weight
    weight: Optional[FloatTensor] = None

    @property
    def realistic(self) -> FloatTensor:
        """Return the realistic rank, i.e. the average of optimistic and pessimistic rank."""
        return 0.5 * (self.optimistic + self.pessimistic).float()

    def __post_init__(self) -> None:
        """Error checking."""
        assert (self.optimistic > 0).all()
        assert (self.pessimistic > 0).all()


def compute_ranks_from_scores(scores: FloatTensor, positive_scores: FloatTensor, easy_targets: LongTensor) -> _Ranks:
    """Compute ranks from scores."""
    # First filter out easy targets
    filtered_scores = scores.clone().float()
    filtered_scores[easy_targets] = -float('inf')
    
    # Debug score distribution for first target (10030)
    
    best_rank = 1 + (filtered_scores.unsqueeze(1) > positive_scores).sum(dim=0)
    worst_rank = (filtered_scores.unsqueeze(1) >= positive_scores).sum(dim=0)

    ranking = torch.argsort(filtered_scores, descending=True)
    
    return _Ranks(optimistic=best_rank, pessimistic=worst_rank, ranking=ranking)


def filter_ranks(
    ranks: LongTensor,
    batch_id: LongTensor,
) -> LongTensor:
    """
    Adjust ranks for filtered setting.

    Determines for each rank, how many smaller ranks there are in the same batch and subtracts this number. Notice that
    this requires that ranks contains all ranks for a certain batch which will be considered for filtering!

    :param ranks: shape: (num_choices,)
        The unfiltered ranks.
    :param batch_id: shape: (num_choices,)
        The batch ID for each rank.

    :return: shape: (num_choices,)
        Filtered ranks.
    """
    smaller_rank = ranks.unsqueeze(dim=0) < ranks.unsqueeze(dim=1)
    same_batch = batch_id.unsqueeze(dim=0) == batch_id.unsqueeze(dim=1)
    adjusted_ranks = ranks - (smaller_rank & same_batch).sum(dim=1)
    assert (adjusted_ranks > 0).all()
    return adjusted_ranks


def score_to_rank_multi_target(scores: FloatTensor, hard_targets: LongTensor, easy_targets: LongTensor) -> _Ranks:
    
    # scores needs to be: (num_entities, batch_size=1)
    scores = scores  # Add batch dimension at end
    
    
    # Create targets tensor
    batch_ids = torch.zeros(1, hard_targets.shape[1], dtype=torch.long)
    entity_ids = hard_targets[0]
    targets = torch.stack((batch_ids.squeeze(0), entity_ids))
    
    
    batch_id, entity_id = targets
    positive_scores = scores[0, entity_id]  # Index with entity_ids, take first batch

    # get unfiltered ranks: shape: (nnz,)
    ranks = compute_ranks_from_scores(
        scores=scores[0], positive_scores=positive_scores, easy_targets=easy_targets[0])

    # First filter using all ranks
    ranks.optimistic = filter_ranks(ranks=ranks.optimistic, batch_id=torch.zeros_like(ranks.optimistic))
    ranks.pessimistic = filter_ranks(ranks=ranks.pessimistic, batch_id=torch.zeros_like(ranks.pessimistic))
    # Then slice to hard targets
    ranks.optimistic_full = ranks.optimistic
    ranks.optimistic = ranks.optimistic[:hard_targets.shape[1]]
    ranks.pessimistic = ranks.pessimistic[:hard_targets.shape[1]]
    # Compute metrics for hard targets only
    # Compute expected rank with hard answers only
    batch_id = batch_id[:hard_targets.shape[1]]

    if hard_targets.shape[1] == 0:
        ranks.weight = None
    else:
        ranks.weight = torch.ones(hard_targets.shape[1], dtype=torch.float32, device=scores.device)

    # expected filtered rank: shape: (nnz,)
    expected_rank = torch.full(
        size=(scores.shape[0],), fill_value=scores.shape[1], device=scores.device)
    expected_rank[batch_id] -= torch.arange(hard_targets.shape[1], device=scores.device)
    expected_rank = 0.5 * (1 + 1 + expected_rank.float())
    expected_rank = expected_rank[batch_id]
    ranks.expected_rank = expected_rank
    return ranks

class ScoreAggregator:
    """An aggregator for scores."""

    @abstractmethod
    def process_scores_(
        self,
        scores: torch.FloatTensor,
        hard_targets: torch.LongTensor,
        easy_targets: torch.LongTensor
    ) -> None:
        """
        Process a batch of scores.

        Updates internal accumulator of ranks.

        :param scores: shape: (batch_size, num_choices)
            The scores for each batch element.
        :param hard_targets: shape: (2, nnz)
            The answer entities, in format (batch_id, entity_id) that cannot be obtained with traversal
        :param easy_targets: shape: (2, nnz)
            The answer entities, in format (batch_id, entity_id) that can be obtained with traversal
        """
        raise NotImplementedError

    def finalize(self) -> Metrics:
        """
        Finalize aggregation and extract result.

        :return:
            A mapping from metric names to the scalar metric values.
        """
        raise NotImplementedError


def _weighted_mean(
    tensor: torch.Tensor,
    weight: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Compute weighted mean.

    :param tensor:
        The tensor.
    :param weight:
        An optional weight.

    :return:
        The (weighted) mean. If weight is None, uniform weights are assumed.
    """
    tensor = tensor.float()
    if weight is None:
        return tensor.mean()
    return (tensor * weight).sum() / weight.sum()


class _RankingMetricAggregator:
    """An aggregator for fixed rank-type ranking metrics."""

    def __init__(
        self,
        ks: Collection[int] = (1, 3, 5, 10),
    ):
        self.ks = ks
        self._data: List[Tuple[Union[int, float], ...]] = []
        self._num_queries = 0  # Add counter for number of queries

    def process_ranks(self, ranks: torch.Tensor, weight: Optional[torch.Tensor]) -> None:
        """Process a tensor of ranks, with optional weights."""
        assert (ranks > 0).all()
        self._num_queries += 1
        
        # Calculate MRR for each hard answer in this query
        mrr_per_answer = ranks.float().reciprocal()
        query_mrr = mrr_per_answer.mean().item()

        self._data.append((
            ranks.shape[0],  # number of targets in this query
            ranks.float().mean().item(),  # mean rank - convert to float first
            query_mrr,  # mean of individual MRRs
            *(
                (ranks <= k).float().mean().item()
                for k in self.ks
            ),
        ))

    def finalize(self) -> Mapping[str, float]:
        """Aggregates ranks into various single-figure metrics."""
        df = pandas.DataFrame(self._data, columns=[
            "batch_size",
            "mean_rank",
            "mean_reciprocal_rank",
            *(
                f"hits_at_{k}"
                for k in self.ks
            ),
        ])
        result = dict(
            num_ranks=self._num_queries,  # Return number of queries instead of total ranks
        )
        # Average metrics across queries
        for column in df.columns:
            if column == "batch_size":
                continue
            value = df[column].mean()  # Simple mean across queries
            if numpy.issubdtype(value.dtype, numpy.integer):
                value = int(value)
            elif numpy.issubdtype(value.dtype, numpy.floating):
                value = float(value)
            result[column] = value
        return result


class RankingMetricAggregator(ScoreAggregator):
    """An aggregator for ranking metrics."""

    def __init__(
        self,
        ks: Collection[int] = (1, 3, 5, 10),
        average: str = MICRO_AVERAGE,
    ):
        """
        Initialize the aggregator.

        :param ks:
            The values for which to compute Hits@k.
        :param average:
            The average mode to use for computing aggregated metrics.
        """
        self.ks = ks
        self.average = average
        self._aggregators: MutableMapping[str, _RankingMetricAggregator] = {
            rank_type: _RankingMetricAggregator(ks=ks)
            for rank_type in RANK_TYPES
        }
        self._expected_ranks: List[torch.Tensor] = []
        self._expected_ranks_weights: List[Optional[torch.Tensor]] = []
        self.per_query_metrics: List[Dict[str, float]] = []
        self.ranks_dict: Dict[str, List[float]] = {}

    def _store_ranks(self, query: QueryGraphBatch, ranks: torch.Tensor) -> None:
        key = get_key_for_query(query)
        self.ranks_dict[key] = ranks.tolist()
        
    @torch.no_grad()
    def process_scores_(
        self,
        query: QueryGraphBatch,
        scores: FloatTensor,
        hard_targets: LongTensor,
        easy_targets: LongTensor,
    ) -> None:  # noqa: D102
        if not torch.isfinite(scores).all():
            raise RuntimeError(f"Non-finite scores: {scores}")

        ranks = score_to_rank_multi_target(
            scores=scores,
            hard_targets=hard_targets,
            easy_targets=easy_targets,
        )

        rank_reciprocal = ranks.realistic.float().reciprocal()

        self._store_ranks(query, ranks.ranking)

        self._aggregators[RANK_OPTIMISTIC].process_ranks(
            ranks.optimistic, weight=ranks.weight)
        self._aggregators[RANK_PESSIMISTIC].process_ranks(
            ranks.pessimistic, weight=ranks.weight)
        self._aggregators[RANK_REALISTIC].process_ranks(
            ranks.realistic, weight=ranks.weight)
        assert ranks.expected_rank is not None
        self._expected_ranks.append(ranks.expected_rank.detach().cpu())
        if ranks.weight is not None:
            self._expected_ranks_weights.append(ranks.weight.detach().cpu())

    def finalize(self) -> Metrics:  # noqa: D102
        result: dict[str, float] = dict()
        for rank_type, agg in self._aggregators.items():
            # Remove "RankType.RANK_" prefix from rank_type
            clean_type = rank_type.replace("RankType.RANK_", "").lower()
            for key, value in agg.finalize().items():
                result[f"{clean_type}.{key}"] = value
        self._aggregators.clear()
        
        # adjusted mean rank (index)
        if len(self._expected_ranks_weights) == 0 or any(w is None for w in self._expected_ranks_weights):
            weights = None
        else:
            weights = torch.cat(cast(List[torch.Tensor], self._expected_ranks_weights))
        expected_mean_rank = _weighted_mean(tensor=torch.cat(self._expected_ranks), weight=weights).item()
        
        # Use clean type for realistic metrics too
        clean_realistic = RANK_REALISTIC.replace("RankType.RANK_", "").lower()
        result[f"{clean_realistic}.expected_mean_rank"] = expected_mean_rank
        result[f"{clean_realistic}.adjusted_mean_rank"] = result[f"{clean_realistic}.mean_rank"] / expected_mean_rank
        result[f"{clean_realistic}.adjusted_mean_rank_index"] = 1 - ((result[f"{clean_realistic}.mean_rank"] - 1) / (expected_mean_rank - 1))

        metrics: Metrics = Metrics(result)
        return metrics
