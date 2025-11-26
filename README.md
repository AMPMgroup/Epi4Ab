**Epi4Ab: Prediction of conformational epitopes for specific antibody VH/VL families and CDRs sequences**
==============
Last updating: `2025-11-26`

# USAGE

## Set up

In Unix terminal, change directory to `Epi4Ab` folder.
```bash
$ cd path/to/Epi4Ab
```

Create virtual environmnet
```bash
$ python -m venv venv
```

Run in virtual environment to ensure consistent version of modules.
```bash
$ source venv/bin/activate
```

Install modules (ignore if already installed).
```bash
$ pip install -r requirements.txt
```

Exit virtual environment using command.
```bash
$ deactivate
```

Create `.env` file in `/Epi4Ab` to contain environmental variables.
```
# User may change the directories here
CURRENT_WORKING_DIRECTORY="path/to/Epi4Ab"
DIRECTORY_INPUT="${CURRENT_WORKING_DIRECTORY}/input"
DIRECTORY_PREPROCESS_OUTPUT="${CURRENT_WORKING_DIRECTORY}/output_preprocess"
# Tool directories
PYMOL_EXECUTABLE="path/to/pymol"
MAFFT_EXECUTABLE="path/to/mafft"


# preprocessing directories
DIRECTORY_PDB_INFO="${DIRECTORY_INPUT}/pdb_info.csv"
DIRECTORY_PROCESSED_DATA="${DIRECTORY_PREPROCESS_OUTPUT}/processed_data"
DIRECTORY_NODES_EDGES="${DIRECTORY_PREPROCESS_OUTPUT}/nodes_edges"
DIRECTORY_METADATA="${DIRECTORY_PREPROCESS_OUTPUT}/metadata.csv"

## training directories
# DIRECTORY_TRAINING_INPUT="${DIRECTORY_INPUT}/train3A.txt"
# DIRECTORY_TESTING_INPUT="${DIRECTORY_INPUT}/test3A.txt"
# DIRECTORY_TESTING_OUTPUT="${CURRENT_WORKING_DIRECTORY}/output"
## In case using AlphaFold model for training/testing
# DIRECTORY_TRAINING_ALPHAFOLD_INPUT="${DIRECTORY_INPUT}/pdb_af.txt"
# DIRECTORY_TESTING_ALPHAFOLD_INPUT="${DIRECTORY_INPUT}/unseen_af.txt"

# inference directories
DIRECTORY_INFERENCE_OUTPUT="${CURRENT_WORKING_DIRECTORY}/output_inference"
DIRECTORY_INFERENCE_PDB_LIST="${DIRECTORY_INPUT}/user_unseen.txt"
FINAL_MODEL_FOLDER="${CURRENT_WORKING_DIRECTORY}/final_trained_Epi4Ab"
```

## PREPROCESS

`pdb_info.csv` (in `Epi4Ab/input`) is required as given format, e.g., containing all required columns.

To preprocess data, run `run_preprocess.sh`.

For batch download from Protein Data Bank (PDB), set `DOWNLOAD_PDB=true` in `run_preprocess.sh`.
Otherwise, do following in `/Epi4Ab`, for example: `1n8z_BAC`.
```bash
mkdir ./output_preprocess
cd ./output_preprocess
mkdir ./processed_data
mkdir ./processed_data/1n8z_BAC
cp path/to/1n8z_chainC.pdb ./processed_data/1n8z_BAC/lig.pdb
```

**NOTE**: 
- Antigen.pdb must be renamed as `lig.pdb` as above.
- Need to download MSMS for biopython to calculate residue depth.
- Need to download MAFFT for sequence alignment.
- Need to download Pymol for nodes, edges creation.

## TRAINING/TESTING

For details, please refer to `source_code/run_setup/arguments.py` or type `--help`.

```bash
$ ./run_training.sh
```

### Output

- Loss result:
    - `train_loss.png`
    - `train_loss.parquet`
- Evaluation:
    - `*_mean_*.txt`: Table of avarage evaluation values
    - `*.parquet`
- Test record: Record of detail prediction of each PDBIDs
- Run log:
    - `log.md`: readable running log.
    - `log.json`: Dictionary type log.
- `model.pt` only saved when running with `--train_all` argument.

## INFERENCE

To perform Epi4Ab inference, use `run_inference.sh`:

`INPUT_AB_FEATURE=false`: Switching to "true" if testing model with different VH/VL and/or CDRs, modifications should be made in `parameters_ab.json`.

```bash
$ ./run_inference.sh
```

### Output
- `test_record`: Record of detail prediction of each PDBIDs.
- `log.md`: readable running log.

## USE trained Epi4Ab model

The trained Epi4Ab model is in `final_trained_Epi4Ab` folder. This model can be used by running `run_inference.sh`. 

**NOTE**: set `FINAL_MODEL_FOLDER="./final_trained_Epi4Ab"` in `.env`

```bash
$ cd path/to/Epi4Ab
$ ./run_inference.sh
```

# REFERENCE
If using Epi4Ab, please cite:

Tran ND, Subramani K, Su CTT. Epi4Ab: a data-driven prediction model of conformational epitopes for specific antibody VH/VL families and CDRs sequences. mAbs 2025, 17(1), p.2531227. [DOI: 10.1080/19420862.2025.2531227](https://doi.org/10.1080/19420862.2025.2531227)
