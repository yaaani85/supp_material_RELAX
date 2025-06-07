import os
import torch
from pathlib import Path
from pytest import fixture
from typing import Annotated

from gqs.loader import QueryGraphBatch
from query_evaluation.engine import execute_pipeline
from query_evaluation.dataset import QEDataset
from query_evaluation.evaluation import Metrics, RankingMetricAggregator
from query_evaluation.ranker import InDegreeRanker
from query_evaluation.dataset import QEDataset

from test_syntentic_dataset import test_syntetic_dataset

@fixture(scope="function")
def run_dataset_test_first() -> None:
    test_syntetic_dataset()

@fixture
def in_degree_ground_truth() -> torch.Tensor:
    # Indegree or: p(O).
    # The dataset (train.nt) used for this test has 8 triples, 6 entities, 3 relations:
    # Neville 0
    # bit 1
    # Harry Potter 1
    # Luna 2
    # Hogwarts 2
    # Weasly 2
    ground_truth = torch.tensor([0/8, 1/8, 1/8, 2/8, 2/8, 2/8])
    return ground_truth

def test_indegree_ranker(in_degree_ground_truth: torch.Tensor,
                         run_dataset_test_first: Annotated[str, fixture]) -> None:

    dataset: QEDataset = QEDataset(dataset_name="syntetic_dataset_for_testing_frequencies",
                                   root_directory=Path(os.getcwd() + "/tests/datasets/"))
    frequency_ranker = InDegreeRanker(dataset)

    test_queries = dataset.test_queries()
    query_batch = test_queries[0]

    scores: torch.Tensor = frequency_ranker.rank(query_batch)
    sorted_scores, _ = torch.sort(scores)
    sorted_ground_truth, _ = torch.sort(in_degree_ground_truth)

    print("Scores ranker", sorted_scores)
    print("Scores ground truth", sorted_ground_truth)
    assert torch.equal(sorted_scores, sorted_ground_truth)
