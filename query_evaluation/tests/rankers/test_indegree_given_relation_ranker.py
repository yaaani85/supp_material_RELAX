import os
import torch
from pathlib import Path
from pytest import fixture
from typing import Annotated

from query_evaluation.engine import execute_pipeline
from query_evaluation.dataset import QEDataset
from query_evaluation.evaluation import Metrics, RankingMetricAggregator
from query_evaluation.ranker import IndegreeGivenRelationRanker

from test_syntentic_dataset import test_syntetic_dataset

@fixture(scope="function")
def run_dataset_test_first() -> None:
    test_syntetic_dataset()

@fixture
def in_degree_ground_truth() -> torch.Tensor:
    # The dataset (train.nt) used for this test has 8 triples, 6 entities, 3 relations:
    # Relations:
    # 0 = alumni, 1 = knows, 2 = logo
    # count per relation
    # Neville (0, 0)
    # bit (2,1)
    # Harry Potter (1,1)
    # Luna (1,2)
    # Hogwarts (0,2)
    # Weasly (1,2)
    ground_truth_per_relation = {0: torch.tensor([0, 0, 0, 0, 0, 1]),
                                 1: torch.tensor([0, 1/5, 2/5, 0/5, 2/5, 0]),
                                 2: torch.tensor([1, 0, 0, 0, 0, 0])}

    return ground_truth_per_relation

def test_indegree_ranker(in_degree_ground_truth: torch.Tensor,
                         run_dataset_test_first: Annotated[str, fixture]) -> None:

    dataset: QEDataset = QEDataset(dataset_name="syntetic_dataset_for_testing_frequencies",
                                   root_directory=Path(os.getcwd() + "/tests/datasets/"))
    frequency_ranker = IndegreeGivenRelationRanker(dataset)

    for query in dataset.test_queries():

        print(query.edge_type)
    #     scores = torch.cat(scores_per_relation, dim=1).mean(dim=1)
    #     return scores
    #     scores: torch.Tensor = frequency_ranker.rank(query)
    #     sorted_scores, _ = torch.sort(scores)
    #     sorted_ground_truth, _ = torch.sort(in_degree_ground_truth[query.edge_)
    #
    # print("Scores ranker", sorted_scores)
    # print("Scores ground truth", sorted_ground_truth)
    # assert torch.equal(sorted_scores, sorted_ground_truth)
