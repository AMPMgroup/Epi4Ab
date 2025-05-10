**Epi4Ab: Prediction of conformational epitopes for specific antibody VH/VL families and CDR H3/L1 sequences**
==============
Last updating: `2025-03-13`

# USAGE

## Set up

In Unix terminal, change directory to `interface_module` folder.
```bash
$ cd path/to/interface_module
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

Create `.env` file to contain environmental variables.
```
DIRECTORY_DATA="/path/to/nodes_edges_folder/"
DIRECTORY_PROCESSED_DATA="/path/to/processed_data_folder/"
DIRECTORY_OUTPUT="/path/to/output/" # In case run_model, set this path to folder of model
DIRECTORY_UNSEEN_OUTPUT="/path/to/run_model_output/"
DIRECTORY_PDB_LIST="/path/to/train_pdb_file/"
DIRECTORY_UNSEEN_PDB_LIST="/path/to/test_pdb_file/"
```
## TRAINING model
### Run script
For the explanation and usage of any argument, please refer to `--help` or `interface_prediction/arguments.py`.

Before training model, make sure that all the directory variables in `.env` are assigned.
```bash
$ ./run_epi.sh
```

### Output
Output of training include
- Loss result: Recorded loss during training
    - `train_loss.png`: Line plot.
    - `train_loss.parquet`: Saved data.

- Evaluation: Prediction result
    - `*_mean_*.txt`: Table of avarage evaluation value.
    - `*.parquet`: Saved data.
- Test record: Record of detail prediction of each PDBIDs.
- Run log:
    - `log.md`: readable running log.
    - `log.json`: Dictionary type log.
- Model: `model.pt` only saved when running with `--train_all` argument.

## PREDICTION

### Run script
Similar to training, test model could be run by modifying these arguments in `run_model.sh`:
- `MODEL_NAME`: Name of folder contains target `model.pt`.
- `USE_REGION`: Choose `true` if test regional data.
- `PLOT_NETWORK`: Choose `true` if want to plot network for test pdbs.
- `INPUT_AB_FEATURE`: Choose `true` it want to specify Antibody feature. The modification should be made in `parameters_ab.json`.

```bash
$ ./run_model.sh
```

### Output
Output of testing includes:
- Evaluation: Prediction result
    - `*_mean_*.txt`: Table of average evaluation value.
    - `*.parquet`: Saved data.
- Test record: 
    - Record of detail prediction of each PDBIDs.
    - Networkx graph if use `PLOT_NETWORK`.
- Run log:
    - `log.md`: readable running log.


