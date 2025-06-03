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
python epi_graph.py $PARAMETERS $DIRECTORY_DATA $DIRECTORY_PROCESSED_DATA $DIRECTORY_PDB_LIST $DIRECTORY_UNSEEN_PDB_LIST \
                    $DIRECTORY_AF_PDB_LIST $DIRECTORY_UNSEEN_AF_PDB_LIST $DIRECTORY_OUTPUT $PARAMS