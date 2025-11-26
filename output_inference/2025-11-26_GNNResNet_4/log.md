# AMPM GROUP - Graph Interface Prediction

Operating date: 2025-11-26

Code version: 2024-12-30

## Seed
| Variable | Value |
| --- | --- |
| Torch | 1 |
| Sklearn | 1 |
| NetworkX | 1 |

## Data
| Variable | Value |
| --- | --- |
| Feature version | v1.0.2 |
| Continuous features | ['resDepth', 'caDepth', 'psi', 'phi', 'omega', 'chi', 'aac', 'cc', 'H1_len', 'H2_len', 'H3_len', 'L1_len', 'L2_len', 'L3_len', 'H3_score', 'L1_score'] |
| One-hot features | ['angleNan', 'chiNan', 'VH1', 'VH2', 'VH3', 'VH4', 'VH5', 'VH6', 'VH7', 'VH8', 'VH9', 'VH14', 'VH_unk', 'VK1', 'VK2', 'VK3', 'VK4', 'VK5', 'VK6', 'VK8', 'VK10', 'VK12', 'VK13', 'VK14', 'VK_others', 'VLa1', 'VLa2', 'VLa3', 'VLa6', 'VLa_others', 'VL_unk'] |
| Amino acid list | ['A', 'V', 'L', 'I', 'P', 'M', 'F', 'W', 'G', 'S', 'T', 'C', 'N', 'Q', 'Y', 'K', 'R', 'H', 'D', 'E'] |
| Use Region | False |
| Use Relaxed data | False |
| Use Alpha Fold data | True |
| Use pre-trained | True |
| Pre-trained model name | ESM2_t30 |
| Pre-trained dim | 640 |
| Freeze pre-trained | True |
| Use pre-trained feed forward | False |
| Pre-trained feed forward dimension | 64 |
| Pre-trained feed forward output | 32 |
| pre-trained feed forward dropout | 0.2 |
| Label | [0, 1, 2] |
| Relation type | Distance |
| Bias distance | 0 |
| Edge threshold | 10.0 |
| Use MHA on | seq |
| MHA head | 4 |
| MHA number of layers | 6 |
| MHA dropout | 0.2 |
| Max antigen length | 670 |
| Use AntiBERTy | True |
| CDRs for AntiBERTy | ['H1', 'H2', 'H3', 'L1', 'L2', 'L3'] |
| Max length for AntiBERTy | 124 |
| Max length dict for AntiBERTy | {'H1': 20, 'H2': 22, 'H3': 30, 'L1': 20, 'L2': 16, 'L3': 16} |
| H3 max length for AntiBERTy | 124 |
| AntiBERTy feed forward dimension | 128 |
| AntiBERTy feed forward output | 16 |
| AntiBERTy feed forward dropout | 0.2 |
| Softmax data | False |
| Data normalization before training | True |
| ElliPro label | PI |
| Error PDB | [] |
## Model
| Variable | Value |
| --- | --- |
| Use base model | False |
| Model code | GNNResNet |
| Model description | None |
| Softmax output | False |
| Number of input features | 687 |
| Number of hidden channels | [128, 64, 96, 128, 96, 96, 32, 64] |
| Number of out labels | 3 |
| Filter size | [5] |
| Attention heads | [5] |
| Drop-out | [0.2, 0.2, 0.0, 0.3, 0.4, 0.3, 0.0, 0.0] |
| Using GAT in Graph | True |
| Number of layers | 8 |
| Dropout edge probability | 0.2 |
| Data weight in initial process | {'pre-trained': 1, 'struct': 1, 'antiberty': 1, 'token': 1} |
| GAT concat | False |
| Block norm | batchnorm |
| Block norm eps | 1e-05 |
| Block norm momentum | 0.1 |
| Use Deep & Shallow | False |
| Shallow layer | 0 |
| Shallow cut-off | 2.0 |
| resDepth index | None |


Operators: Cheb - (Block) x num_layers - (out layer)

Activation: LeakyReLU

- 01:
    - Block: Cheb
    - out layer: GAT
- 02:
    - Block: Cheb GAT
    - out layer: Linear


## Optimizer
| Variable | Value |
| --- | --- |
| Optimizer method | adam |
| Learning rate | 0.0001 |
| Weight decay | 0.0016 |
| Momentum | 0.9 |
        
## Loss function
| Variable | Value |
| --- | --- |
| Loss function | cross_entropy |
| Weight of cross entropy | [10.0, 35.0, 20.0] |

## Train
| Variable | Value |
| --- | --- |
| Number of epoches | 2000 |
| Batch size | 44 |
| Train all data | yes |
| Gradient Attribute | False |
| Attribute weight | [0.1735178381204605, -0.05394153669476509, -0.0020570196211338043] |
| Include bond potential | True |
| Include Lennard-Jones potential | True |
| Include charge potential | True |

## Evaluation method
| Variable | Value |
| --- | --- |
| Methods | ['Recall', 'Precision', 'f1', 'Accuracy', 'ROC AUC', 'Average Precision'] |
## Directory
| Variable | Value |
| --- | --- |
| Data | /mnt/edward_data/ampm_project/vsc_module/Epi4Ab/output_preprocess/nodes_edges |
| PDB list | /mnt/edward_data/ampm_project/vsc_module/Epi4Ab/input/user_unseen.txt |
| Output | /mnt/edward_data/ampm_project/vsc_module/Epi4Ab/output_inference |

## Runtime
| Variable | Value |
| --- | --- |
| Preparing | 0:00:00.464374 |
| Processing data | 0:03:29.184956 |
| Generating result | 0:00:11.354621 |
| Device | cuda |

## Saving model
| Variable | Value |
| --- | --- |
| Saving | True |
| State dictionary | True |