**Epi4Ab: Prediction of conformational epitopes for specific antibody VH/VL families and CDRs sequences**
==============
Last updating: `2025-11-26`

# USAGE

## Set up

In Unix terminal, change directory to `Epi4Ab` folder, and do the following:
```bash
cd path/to/Epi4Ab

# Create virtual environmnet
python -m venv venv

source venv/bin/activate

# Install modules (ignore if already installed).
pip install -r requirements.txt

# to exit the environment
deactivate
```



Create `.env` file in `/Epi4Ab` to contain required variables:
```
# Please modify paths accordingly

CURRENT_WORKING_DIRECTORY="path/to/Epi4Ab"
DIRECTORY_INPUT="${CURRENT_WORKING_DIRECTORY}/input"
DIRECTORY_PREPROCESS_OUTPUT="${CURRENT_WORKING_DIRECTORY}/output_preprocess"

# Need PyMOL and MAFFT
PYMOL_EXECUTABLE="path/to/pymol"
MAFFT_EXECUTABLE="path/to/mafft"

# preprocessing
DIRECTORY_PDB_INFO="${DIRECTORY_INPUT}/pdb_info.csv"
DIRECTORY_PROCESSED_DATA="${DIRECTORY_PREPROCESS_OUTPUT}/processed_data"
DIRECTORY_NODES_EDGES="${DIRECTORY_PREPROCESS_OUTPUT}/nodes_edges"
DIRECTORY_METADATA="${DIRECTORY_PREPROCESS_OUTPUT}/metadata.csv"

# In case, re-training Epi4Ab model
# DIRECTORY_TRAINING_INPUT="${DIRECTORY_INPUT}/train.txt"
# DIRECTORY_TESTING_INPUT="${DIRECTORY_INPUT}/test.txt"
# DIRECTORY_TESTING_OUTPUT="${CURRENT_WORKING_DIRECTORY}/output"

# In case using additional AlphaFold models for training/testing
# DIRECTORY_TRAINING_ALPHAFOLD_INPUT="${DIRECTORY_INPUT}/pdb_af.txt"
# DIRECTORY_TESTING_ALPHAFOLD_INPUT="${DIRECTORY_INPUT}/unseen_af.txt"

# inference 
DIRECTORY_INFERENCE_OUTPUT="${CURRENT_WORKING_DIRECTORY}/output_inference"
DIRECTORY_INFERENCE_PDB_LIST="${DIRECTORY_INPUT}/user_unseen.txt"
FINAL_MODEL_FOLDER="${CURRENT_WORKING_DIRECTORY}/final_trained_Epi4Ab"
```

## PREPROCESS
```bash
mkdir ./input
```
`pdb_info.csv` (in `Epi4Ab/input`) is required. 

Please refer `/examples/input/pdb_info.csv`for more details, e.g., containing all required columns.

To preprocess data, execute `run_preprocess.sh`.

For batch download from Protein Data Bank (PDB), set `DOWNLOAD_PDB=true` in `run_preprocess.sh`.
Otherwise, do the following in `/Epi4Ab`, for example: `1n8z_BAC`
```bash
mkdir ./output_preprocess
cd ./output_preprocess
mkdir ./processed_data
mkdir ./processed_data/1n8z_BAC
cp path/to/1n8z_chainC.pdb ./processed_data/1n8z_BAC/lig.pdb
```

**NOTE**: 
- `antigen.pdb` must be renamed as `lig.pdb` as above.
- MSMS is required for biopython to calculate residue depth.
- MAFFT is required for sequence alignment.
- PyMOL is required to create nodes_edges used for graphs


## TRAINING/TESTING

For details of arguments, please refer to ⁠`source_code/run_setup/arguments.py` ⁠or ⁠`--help`

**NOTE**: Need `node_label_pi.parquet` (annotating true labels, e.g. {0,1,2} for each residue_id) for re-training. Methods can be found in *Tran et al. (2025)* below.

Refer `/examples/output_preprocess/nodes_edges` for more details.

```bash
./run_training.sh
```

### Output

-   Loss result:
    - ⁠ `train_loss.png` ⁠
    - ⁠ `train_loss.parquet` ⁠
-   Evaluation:
    - ⁠ `*_mean_*.txt` ⁠: avarage evaluation values
    - ⁠ `*.parquet` ⁠
-   `test record`: detail prediction results of each antigen in `test.txt`
-   Log files:
    - ⁠ `log.md`
    - ⁠ `log.json`
-   `model.pt `⁠only saved when running with ⁠`--train_all`⁠ argument.


## INFERENCE

To perform epitope inference using Epi4Ab, execute `run_inference.sh`:

`INPUT_AB_FEATURE=true`: if users wish to perform quick test with specific VH/VL and/or CDRs for the whole dataset at once, modifications of which should be made in `/Epi4Ab/parameters_ab.json`.

```bash
./run_inference.sh
```

### Output
- `test_record`: prediction results of each antigen in `user_unseen.txt`.
- `log.md`: details of used parameters

## USING trained Epi4Ab model

The trained Epi4Ab model is in `final_trained_Epi4Ab` folder.

**NOTE**: set `FINAL_MODEL_FOLDER="./final_trained_Epi4Ab"` in `.env`

```bash
cd path/to/Epi4Ab
./run_inference.sh
```


# REFERENCE
If using Epi4Ab, please cite:

Tran ND, Subramani K, Su CTT. Epi4Ab: a data-driven prediction model of conformational epitopes for specific antibody VH/VL families and CDRs sequences. mAbs (2025), 17(1), p.2531227. [doi:10.1080/19420862.2025.2531227](https://doi.org/10.1080/19420862.2025.2531227)

