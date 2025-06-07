#!/bin/bash

set -e

# Array of all query types
QUERY_TYPES=("1p" "2p" "3p" "2i" "3i" "1hop-2i" "2i-1hop")

# Array of datasets
DATASETS=("nellcqahard2" "fcqahard2" "icecqahard2" "fcqa" "nellcqa")

# Ranker type (you can modify this)
RANKER="combined"

# Number of queries (optional, remove --max-queries if you want all)
MAX_QUERIES=1

# Base directory for results
RESULTS_DIR="results"

# Loop through each dataset
for dataset in "${DATASETS[@]}"
do
    # Loop through each query type
    for query_type in "${QUERY_TYPES[@]}"
    do
        echo "Evaluating dataset: $dataset, query type: $query_type"
        
        # Create directories for results
        mkdir -p "$RESULTS_DIR/$dataset/$query_type"
        mkdir -p "$RESULTS_DIR/$query_type"
        
        query_evaluation evaluate \
            --dataset $dataset \
            --query-type $query_type \
            --ranker $RANKER \
            --tie-breakers "indegree-relation" \
            --tie-breakers "indegree" \
            --write-to-file \
            --output-file-path "$RESULTS_DIR/$dataset/$query_type/results_$RANKER.json" 
    done
done
