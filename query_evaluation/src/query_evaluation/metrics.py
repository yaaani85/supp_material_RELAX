"""Evaluation utilities."""
from __future__ import annotations
import dataclasses
import logging
from abc import abstractmethod

# from .data.mapping import get_entity_mapper
from query_evaluation.custom_types import FloatTensor, LongTensor

__all__ = [
    "Metrics"
]

logger = logging.getLogger(__name__)

MICRO_AVERAGE = "micro"
MACRO_AVERAGE = "macro"

@dataclasses.dataclass
class Metrics:
    def __init__(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to a dictionary format."""
        return self.metrics

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return self.__str__()

    def __str__(self):
        # Group metrics by their prefix (everything before the dot)
        grouped_metrics = {}
        max_length = 0
        
        for key, value in self.metrics.items():
            if "." in key:
                prefix, metric = key.split(".")
                if prefix not in grouped_metrics:
                    grouped_metrics[prefix] = {}
                grouped_metrics[prefix][metric] = value
                max_length = max(max_length, len(metric))
            else:
                # Handle metrics without a prefix
                if "general" not in grouped_metrics:
                    grouped_metrics["general"] = {}
                grouped_metrics["general"][key] = value
                max_length = max(max_length, len(key))

        # Format the output
        result = []
        for group_name, group_metrics in grouped_metrics.items():
            result.append(f"\n{group_name}:")
            for metric, value in group_metrics.items():
                padding = " " * (max_length - len(metric))
                result.append(f"\t{metric}{padding} = {value}")

        return "\n".join(result)
