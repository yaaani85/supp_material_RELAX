from pathlib import Path
from query_evaluation.dataset import QEDataset
from os import getcwd
from query_evaluation.ranker import OutDegreeRanker

def test_syntetic_dataset() -> None:

    dataset: QEDataset = QEDataset(dataset_name="syntetic_dataset_for_testing_frequencies", root_directory=Path(getcwd() + "/tests/datasets/"))

    assert dataset.entity_mapper.number_of_real_entities() == 6, "number of entities in data set changed"

    assert dataset.relation_mapper.number_of_relation_types() == 3, "number of relation types in  data set changed"

    frequency_ranker = OutDegreeRanker(dataset)
    assert frequency_ranker.graph.get_total_number_of_edges() == 8

    # add more checks
