#!/bin/bash

# Set environment variables
set -o allexport && source .env && set +o allexport

DOWNLOAD_PDB=false

PARAMS=""
[ "$DOWNLOAD_PDB" = true ] && PARAMS+="--download_pdb " || PARAMS+=""


python preprocess/create_metadata.py $DIRECTORY_PDB_INFO $DIRECTORY_PREPROCESS_OUTPUT $MAFFT_EXECUTABLE

python preprocess/main.py $DIRECTORY_METADATA $DIRECTORY_PROCESSED $PARAMS

python preprocess/nodes_edges.py $DIRECTORY_METADATA $DIRECTORY_PROCESSED $DIRECTORY_NODES_EDGES $PYMOL_EXECUTABLE 'CB'
python preprocess/nodes_edges.py $DIRECTORY_METADATA $DIRECTORY_PROCESSED $DIRECTORY_NODES_EDGES $PYMOL_EXECUTABLE 'CA'

python preprocess/fill_edge.py $DIRECTORY_METADATA $DIRECTORY_NODES_EDGES