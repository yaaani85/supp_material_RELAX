import click
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from query_evaluation.engine import QueryEvaluationEngine
from query_evaluation.dataset import QEDataset
from query_evaluation.custom_types import RankerType
from query_evaluation.utils import _get_output_path_suffix, save_results_to_file
from query_evaluation.factory import get_ranker

@click.group(context_settings={'show_default': True})
@click.option(
    '--log',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False),
    default='INFO',
    help='Set the logging level',
    show_default=True,
)
def main(log: str) -> None:
    """The main entry point."""
    # Load environment variables first
    load_dotenv()
    
    # Configure logging based on command line option
    logging.basicConfig(
        level=getattr(logging, log.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(  # File output
                filename=Path('query_evaluation.log'),
                mode='a'
            )
        ]
    )
    
    # Log initial configuration
    logging.info(f"Starting query evaluation with log level: {log}")


@main.command(help="Evaluate a query answering dataset")
@click.option(
    "--ranker",
    type=click.Choice([r.value for r in RankerType]),
    help="The primary ranker to use for evaluation",
    required=True,
)
@click.option(
    "--primary",
    type=click.Choice([r.value for r in RankerType]),
    help="Optional primary ranker for hybrid/combined rankers (defaults to relaxed-anchor)",
    default=None,
)
@click.option(
    "--tie-breakers",
    type=click.Choice([r.value for r in RankerType]),
    help="Optional tie-breaking rankers (applied in order)",
    multiple=True,
    default=(),
)
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="Path to the datasets directory",
)
@click.option(
    "--query-type",
    type=click.Choice(['1hop', '2hop', '3hop', '2i', '3i', '2i-1hop', '1hop-2i', 'all']),
    default='all',
    help="Specific query type to evaluate. Will look in 0qual directory.",
    show_default=True,
)
@click.option(
    "--output-file-path",
    type=str,
    default="results.json",
    help="Path to save the evaluation results",
    show_default=True,
)
@click.option(
    "--max-queries",
    type=int,
    default=None,
    help="Maximum number of queries to evaluate. Default is all queries.",
    show_default=True,
)
@click.option(
    "--write-to-file",
    is_flag=True,
    help="Write per-query metrics to files for each ranker",
    show_default=True,
)
def evaluate(
    dataset: str, 
    ranker: str, 
    primary: str,
    tie_breakers: tuple[str, ...],
    query_type: str, 
    output_file_path: str, 
    max_queries: int, 
    write_to_file: bool
) -> None:
    """Evaluate a query answering dataset"""
    repository_id = dataset
    
    # Get output path with appropriate suffixes
    output_file_path = _get_output_path_suffix(
        output_file_path, ranker, tie_breakers, query_type, repository_id
    )
    
    logging.info(f"Evaluating dataset: {dataset}")
    qe_dataset = QEDataset(dataset, query_type=query_type)
    
    # Create primary ranker first if specified
    primary_ranker = None
    if primary:
        primary_ranker = get_ranker(primary, qe_dataset, repository_id)
    
    # Initialize ranker(s) using the factory approach
    rankers = [get_ranker(
        ranker,  # primary ranker type
        qe_dataset,
        repository_id,
        tie_breakers=tie_breakers,
        primary_ranker=primary_ranker
    )]
    
    # Create and run evaluation engine
    engine = QueryEvaluationEngine(
        qe_dataset, 
        rankers, 
        max_queries=max_queries, 
        write_to_file=write_to_file
    )
    metrics = engine.evaluate()
    
    save_results_to_file(metrics, output_file_path)
    logging.info(f"Results saved to {output_file_path}")

