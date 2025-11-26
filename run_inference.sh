#!/bin/bash

# Set environment variables
set -o allexport && source .env && set +o allexport

INPUT_AB_FEATURE=false # Switching to "true" if testing model with different VH/VL Ab family

PARAMS=""
[ "$INPUT_AB_FEATURE" = true ] && PARAMS+="--ab_feature_input " || PARAMS+=""

python epi_prediction.py $DIRECTORY_NODES_EDGES $DIRECTORY_PROCESSED_DATA \
    $DIRECTORY_INFERENCE_PDB_LIST $FINAL_MODEL_FOLDER \
    $DIRECTORY_INFERENCE_OUTPUT $PARAMS