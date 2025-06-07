import os
import torch
from typing import Annotated
from pathlib import Path
from pytest import fixture

from query_evaluation.engine import execute_pipeline
from query_evaluation.dataset import QEDataset
from query_evaluation.evaluation import Metrics, RankingMetricAggregator
from query_evaluation.ranker import OutDegreeRanker
from query_evaluation.dataset import QEDataset
from gqs.loader import QueryGraphBatch

from test_syntentic_dataset import test_syntetic_dataset

@fixture(scope="function")
def run_dataset_test_first() -> None:
    test_syntetic_dataset()

@fixture
def out_degree_ground_truth() -> torch.Tensor:
    # Outdegree or p(S)
    # The dataset (train.nt+val.nt) used for this test has 8 triples, 6 entities, 3 relations:
    # Outgoing edges in fake dataset:
    # Neville 3
    # bit 0
    # Harry Potter 3
    # Luna 1
    # Hogwarts 1
    # Weasly 0
    ground_truth = torch.tensor([3/8, 0/8, 3/8, 1/8, 1/8, 0/8])
    return ground_truth


def test_outdegree_ranker(out_degree_ground_truth: torch.Tensor,
                          run_dataset_test_first: Annotated[str, fixture]) -> None:

    dataset: QEDataset = QEDataset(dataset_name="syntetic_dataset_for_testing_frequencies",
                                   root_directory=Path(os.getcwd() + "/tests/datasets/"))
    frequency_ranker = OutDegreeRanker(dataset)

    test_queries = dataset.test_queries()
    query_batch = test_queries[0]

    scores: torch.Tensor = frequency_ranker.rank(query_batch)
    sorted_scores, _ = torch.sort(scores)
    sorted_ground_truth, _ = torch.sort(out_degree_ground_truth)

    print("Scores ranker", sorted_scores)
    print("Scores ground truth", sorted_ground_truth)

    assert torch.equal(sorted_scores, sorted_ground_truth)
