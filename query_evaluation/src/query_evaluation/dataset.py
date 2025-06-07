from typing import Optional
from gqs.dataset import Dataset as GQSDataset
from torch.utils.data import Dataset as TorchDataset
# TODO either make "qet_query_dataset" class method of QEDataset or staticmethod at QQSDataset
from query_evaluation.graph.load import loadTripleFile_non_robust
from gqs.loader import get_query_datasets
from gqs.query_representation.torch import TorchQuery
from gqs.sample import Sample
from query_evaluation.graph.graph import HashVertexGraph
from pathlib import Path

__ALL__ = ["QEDataset"]


class QEDataset(GQSDataset):
    """QueryEvaluation Dataset that extends GQSDataset.
    
    This class provides functionality for working with query evaluation datasets,
    building upon the graph_query_sampler framework.
    
    Args:
        dataset_name: Name of the dataset to load
        query_type: Query type to filter test queries
        root_directory: Optional root directory path for the dataset
    """

    def __init__(self, dataset_name: str, query_type: str = 'all', root_directory: Optional[Path] = None):
        super().__init__(dataset_name, root_directory)
        self._train_val_graph: Optional[HashVertexGraph[str, str]] = None
        self._test_queries: Optional[TorchDataset[TorchQuery]] = None
        self.query_type = query_type
        self.repository_id = dataset_name
        self.num_relations = self.relation_mapper.number_of_relation_types()
        self.num_entities = self.entity_mapper.number_of_real_entities()

    @property
    def train_and_validation_graph(self) -> HashVertexGraph[str, str]:
        """Loads and caches the combined train and validation graph.
        
        Returns:
            HashVertexGraph containing both training and validation triples
        
        Raises:
            FileNotFoundError: If train or validation files cannot be found
        """
        if self._train_val_graph is not None:
            return self._train_val_graph
            
        try:
            with open(self.train_split_location()) as train, \
                 open(self.validation_split_location()) as valid:
                self._train_val_graph = loadTripleFile_non_robust([train, valid])
                return self._train_val_graph
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load dataset files: {e}")

    @property
    def test_queries(self) -> TorchDataset[TorchQuery]:
        """The default test queries dataset.
        
        Returns:
            TorchDataset containing all test queries
        """
        if self._test_queries is None:
            self._test_queries = self._get_test_queries()
        return self._test_queries

    def _get_test_queries(self) -> TorchDataset[TorchQuery]:
        """Retrieves filtered test queries based on the query type."""
        if self.query_type == 'all':
            # Default behavior: all queries
            sample = Sample("**", "*")
        else:
            print(f"Filtering test queries for query type: {self.query_type}")
            # Use query type directly as it matches directory structure
            sample = Sample(f"**/{self.query_type}/*", "*")
        
        query_set, _ = get_query_datasets(self, [], [], [sample])
        return query_set["test"]
