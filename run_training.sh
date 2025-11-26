#!/bin/bash

# Set environment variables
set -o allexport && source .env && set +o allexport

PARAMS=""
while IFS='=' read -r key value || [ -n "$key" ]; do
    [[ ! $key =~ ^\ *# && -n $key ]] && PARAMS+="--${key// /} $value "
done < parameters.txt

PARAMETERS="$(<parameters.txt)"
PARAMETERS=$(echo "$PARAMETERS" | tr '\n' ';')
PARAMETERS=$(echo "$PARAMETERS" | sed 's/ /+/g')
python epi_graph.py $PARAMETERS $DIRECTORY_NODES_EDGES $DIRECTORY_PROCESSED_DATA $DIRECTORY_TRAINING_INPUT $DIRECTORY_TESTING_INPUT \
                    $DIRECTORY_TRAINING_ALPHAFOLD_INPUT $DIRECTORY_TESTING_ALPHAFOLD_INPUT $DIRECTORY_TESTING_OUTPUT $PARAMS