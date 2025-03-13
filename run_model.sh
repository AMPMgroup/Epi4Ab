#!/bin/bash

# Set environment variables
set -o allexport && source .env && set +o allexport

# MODEL_NAME="2024-12-27_GNNNaive_trainALL-FINAL"
MODEL_NAME="2025-02-19_GCNbase_FINAL_batchNorm"
USE_REGION=false
USE_RELAXED=false
PLOT_NETWORK=false
INPUT_AB_FEATURE=false # Use when test same VH/VL Ab family

MODEL_FOLDER=$DIRECTORY_OUTPUT/$MODEL_NAME
PARAMS=""
[ "$USE_REGION" = true ] && PARAMS+="--use_region " || PARAMS+=""
[ "$USE_RELAXED" = true ] && PARAMS+="--use_relaxed " || PARAMS+=""
[ "$PLOT_NETWORK" = true ] && PARAMS+="--plot_network " || PARAMS+=""
[ "$INPUT_AB_FEATURE" = true ] && PARAMS+="--ab_feature_input " || PARAMS+=""

python epi_model.py $DIRECTORY_DATA $DIRECTORY_PROCESSED_DATA $DIRECTORY_UNSEEN_PDB_LIST $MODEL_FOLDER $DIRECTORY_UNSEEN_OUTPUT $PARAMS