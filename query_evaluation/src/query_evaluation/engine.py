from typing import List, Dict
from query_evaluation.evaluation import RankingMetricAggregator
from query_evaluation.metrics import Metrics
from query_evaluation.ranker import Ranker
from query_evaluation.custom_types import RankerType
from query_evaluation.factory import get_ranker
from query_evaluation.dataset import QEDataset
from gqs.loader import QueryGraphBatch
from pathlib import Path
import os
from dotenv import load_dotenv
from tqdm import tqdm
import logging
import torch
import argparse
from query_evaluation.utils import get_key_for_query

class QueryEvaluationEngine:
    def __init__(self, dataset: QEDataset, rankers: List[Ranker], max_queries: int = None, write_to_file: bool = False):
        self.dataset = dataset
        self.test_queries = dataset.test_queries
        self.max_queries = max_queries or len(self.test_queries)
        self.rankers = rankers
        self.metric_aggregators = {
            ranker.__class__.__name__: RankingMetricAggregator()
            for ranker in rankers
        }
        self.write_to_file = write_to_file

    def evaluate(self) -> Dict[str, Metrics]:
        """Execute the evaluation pipeline for all rankers."""
        # hard_answers = 0
        query_type = self.dataset.query_type
        self.scores_dict = {}

        for query_idx, query in tqdm(
            enumerate(self.test_queries), 
            desc="Evaluating queries",
            total=min(self.max_queries, len(self.test_queries))
        ): 
            if query_idx >= self.max_queries:
                break
            try:
                self._process_query(query, query_idx)

            except TimeoutError as e:
                logging.warning(f"Timeout occurred while processing query {query_idx, query.edge_index, query.edge_type}: {e}")
                continue

        results = self._finalize_results()

        
        return results
    
   

    def _process_query(self, query: QueryGraphBatch, query_idx: int) -> None:
        """Process a single query for all rankers."""
        for ranker in self.rankers:
            logging.debug(f"Ranking query {query_idx} with {ranker.__class__.__name__}")
            scores = ranker.rank(query_idx, query)


            aggregator = self.metric_aggregators[ranker.__class__.__name__]
            aggregator.average = "micro"
            aggregator.process_scores_(
                query=query,
                scores=scores.unsqueeze(0),
                hard_targets=query.hard_targets.unsqueeze(0),
                easy_targets=query.easy_targets.unsqueeze(0)
            )

    def _finalize_results(self) -> Dict[str, Metrics]:
        """Compute final metrics for all rankers."""
        results = {}
        for ranker_name, aggregator in self.metric_aggregators.items():
            metrics: Metrics = aggregator.finalize()
            print(f"\nResults for {ranker_name}: {metrics}")
            results[ranker_name] = metrics
            if self.write_to_file:
                self._write_per_query_metrics(ranker_name, aggregator.ranks_dict)
        return results

    def _write_per_query_metrics(self, ranker_name: str, scores_dict: Dict[str, List[float]]) -> None:
        """Write per-query metrics to files for each ranker."""
            # Create directories
        directory = f"results/scores/{self.dataset.repository_id}/{self.dataset.query_type}"
        os.makedirs(directory, exist_ok=True)
        import pickle
            # # Write MRR values (keep existing file unchanged)
        scores_filename = f"{directory}/{ranker_name}_scores.pkl"
        with open(scores_filename, 'wb') as f:  
            pickle.dump(scores_dict, f)
            

    # For quick testing without CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-to-file', action='store_true',
                       help='Write per-query metrics to files')
    # ... add other arguments as needed ...
    
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Configure logging based on environment variable
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Get repository ID from environment variable, fallback to harrypotterdev if not set
    repository_id = os.getenv("GRAPH_DB_REPO_ID", "harrypotterdev")
    print(f"Using repository ID: {repository_id}")
    dataset = QEDataset(repository_id, Path("../../datasets/"))
    rankers = [get_ranker(RankerType.RELAXED_ANCHOR, dataset, repository_id)]
    engine = QueryEvaluationEngine(
        dataset, 
        rankers,
        write_to_file=args.write_to_file
    )
    # metrics = engine.evaluate()
