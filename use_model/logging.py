from datetime import date
import torch
import os
import json
from pathlib import Path
from .change_ab_feature import extract_ab_input_feature

class ModelParamsLogging:
    '''
    This class is for read_model.py to read the parameters of model
    '''
    def __init__(self, log_dict, args):
        assert isinstance(log_dict, dict)
        for key, val in log_dict.items():
            if key not in ['train_time']:
                setattr(self, key, val)
        self.directory_model_folder = args.directory_model_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UseModelLogging:
    def __init__(self, log_dict, args):
        assert isinstance(log_dict, dict)
        for key, val in log_dict.items():
            if key not in ['train_time']:
                setattr(self, key, val)
        self.run_date = date.today()
        # Directory
        self.directory_data = args.directory_data
        self.directory_processed_data = args.directory_processed_data
        self.directory_pdb_list = args.directory_pdb_list
        self.directory_model_folder = args.directory_model_folder
        self.directory_output = args.directory_output
        # Data
        self.use_region = args.use_region
        self.use_relaxed = args.use_relaxed
        self.plot_network = args.plot_network
        # if (self.use_pretrained == 'yes') or (self.use_pretrained == 'combined'):
        #     from transformers import BertModel, BertTokenizer
        #     self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        #     self.bert_model = BertModel.from_pretrained("Rostlab/prot_bert")
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Seed
        self.networkx_seed = args.networkx_seed
        # Antibody input
        self.ab_feature_input = args.ab_feature_input
        if self.ab_feature_input:
            with open(Path(__file__).parent / '../parameters_ab.json', 'r') as f:
                self.ab_feature_dict = json.load(f)
            self.ab_continuous_columns_change, self.ab_continuous_columns_value, self.ab_onehot_vh_columns_value, self.ab_onehot_vl_columns_value = extract_ab_input_feature(self.ab_feature_dict, 
                                                                                                                                                    self.ab_continuous_columns, 
                                                                                                                                                    self.ab_onehot_vh_columns, 
                                                                                                                                                    self.ab_onehot_vl_columns)
    def log_result(self):
        message = f'''# AMPM GROUP - Graph Interface Prediction

Operating date: {self.run_date}

Code version: {self.code_version}

## Seed
| Variable | Value |
| --- | --- |
| Torch | {self.torch_seed} |
| Sklearn | {self.sklearn_seed} |
| NetworkX | {self.networkx_seed} |

## Data
| Variable | Value |
| --- | --- |
| Feature version | {self.feature_version} |
| Continuous features | {self.continuous_columns} |
| One-hot features | {self.onehot_columns} |
| Amino acid list | {self.amino_acid_list} |'''
        if self.use_token:
            message += f'''
| Tokenize data | {self.use_token} |
| Token size | {self.token_size} |
| Token dimension | {self.token_dim} |'''
        if self.feature_version != 'v1.0.2':
            message += f'''   
| Using struct | {self.use_struct} |'''

        message += f'''
| Use Region | {self.use_region} |
| Use Relaxed data | {self.use_relaxed} |
| Use Alpha Fold data | {self.use_alphafold} |
| Use pre-trained | {self.use_pretrained} |'''
        if self.use_pretrained:
            message += f'''
| Pre-trained model name | {self.pretrained_model} |
| Pre-trained dim | {self.pretrained_dim} |
| Freeze pre-trained | {self.freeze_pretrained} |
| Use pre-trained feed forward | {self.use_seq_ff} |
| Pre-trained feed forward dimension | {self.seq_ff_dim} |
| Pre-trained feed forward output | {self.seq_ff_out} |
| pre-trained feed forward dropout | {self.seq_ff_dropout} |'''

        if self.loss_function == 'mse':
            message += f'''
| Label | Continuous |'''
        else:
            message += f'''
| Label | {[i for i in range(self.out_label)]} |'''
        if self.edge_type == 'dist':
            message += f'''
| Relation type | Distance |
| Bias distance | {self.bias_distance} |
| Edge threshold | {self.edge_threshold} |'''
        elif self.edge_type == 'comm':
            message += f'''
| Relation type | Communication |'''
        else:
            message += f'''
| Relation type | Communication / (Distance - 2) |'''

        message += f'''
| Use MHA on | {self.use_mha_on} |
| MHA head | {self.mha_head} |
| MHA number of layers | {self.mha_num_layers} |
| MHA dropout | {self.mha_dropout} |
| Max antigen length | {self.max_antigen_len} |
| Use AntiBERTy | {self.use_antiberty} |
| CDRs for AntiBERTy | {self.antiberty_cdr_list} |
| Max length for AntiBERTy | {self.antiberty_max_len} |
| Max length dict for AntiBERTy | {self.antiberty_max_len_dict} |
| H3 max length for AntiBERTy | {self.antiberty_max_len} |
| AntiBERTy feed forward dimension | {self.antiberty_ff_dim} |
| AntiBERTy feed forward output | {self.antiberty_ff_out} |
| AntiBERTy feed forward dropout | {self.antiberty_ff_dropout} |
| Softmax data | {self.softmax_data} |
| Data normalization before training | {not self.not_normalize_data} |
| ElliPro label | {'PI' if self.label_from_ellipro_pi else 'Linear'} |
| Error PDB | {self.error_data_process_list} |'''

        message += f'''
## Model
| Variable | Value |
| --- | --- |
| Use base model | {self.use_base_model} |
| Model code | {self.model_name} |
| Model description | {self.model_description} |
| Softmax output | {self.softmax_output} |
| Number of input features | {self.in_feature} |
| Number of hidden channels | {self.hidden_channel} |
| Number of out labels | {self.out_label} |
| Filter size | {self.filter_size} |
| Attention heads | {self.attention_head} |
| Drop-out | {self.drop_out} |
| Using GAT in Graph | {not self.not_include_gat} |
| Number of layers | {self.num_layers} |
| Dropout edge probability | {self.dropout_edge_p} |
| Data weight in initial process | {self.initial_process_weight_dict} |
| GAT concat | {self.gat_concat} |
| Block norm | {self.block_norm} |'''
        if self.block_norm:
            message += f'''
| Block norm eps | {self.block_norm_eps} |
| Block norm momentum | {self.block_norm_momentum} |
| Use Deep & Shallow | {self.use_deep_shallow} |
| Shallow layer | {self.shallow_layer} |
| Shallow cut-off | {self.shallow_cutoff} |
| resDepth index | {self.resDepth_index} |'''
        message += f'''

{self.model_architecture}

## Optimizer
| Variable | Value |
| --- | --- |
| Optimizer method | {self.optimizer_method} |
| Learning rate | {self.learning_rate} |
| Weight decay | {self.weight_decay} |
| Momentum | {self.momentum} |'''
            
        message += f'''
        
## Loss function
| Variable | Value |
| --- | --- |
| Loss function | {self.loss_function} |'''
        if self.loss_function == 'mse':
            message += f'''
| Threshold of MSE | {self.mse_threshold} |'''
        elif self.loss_function == 'cross_entropy':
            message += f'''
| Weight of cross entropy | {self.cross_entropy_weight} |'''
        elif self.loss_function == 'hce':
            message += f'''
| Weight of cross entropy | {self.cross_entropy_weight} |'''

        message += f'''

## Train
| Variable | Value |
| --- | --- |
| Number of epoches | {self.epoch_number} |
| Batch size | {self.batch_size} |
| Train all data | {self.train_all} |'''
        if self.edge_type == 'dist':
            message += f'''
| Gradient Attribute | {self.gradient_attribute} |'''
            if self.gradient_attribute:
                message += f'''
| Gradient Attribute weight | {self.gradient_attribute_weight} |'''
                if self.gradient_attribute_with_bias:
                    message += f'''
| Gradient Attribute bias | {self.gradient_attribute_bias} |'''
                else:
                    message += f'''
| Gradient Attribute bias | No |'''
            else:
                message += f'''
| Attribute weight | {self.attribute_weight} |'''
                    
            message += f'''
| Include bond potential | {not self.attribute_no_bond} |
| Include Lennard-Jones potential | {not self.attribute_no_lj} |
| Include charge potential | {not self.attribute_no_charge} |'''
        message += f'''

## Evaluation method
| Variable | Value |
| --- | --- |
| Methods | {self.evaluation_columns} |'''

        if self.ab_feature_input:
            message += f'''
| Continuous  | {self.ab_feature_dict['continuous']} |
| VH family | {self.ab_feature_dict['vh_type']} |
| VH family | {self.ab_feature_dict['vl_type']} |'''
            
        message += f'''
## Directory
| Variable | Value |
| --- | --- |
| Data | {self.directory_data} |
| PDB list | {self.directory_pdb_list} |
| Output | {self.directory_output} |

## Runtime
| Variable | Value |
| --- | --- |
| Preparing | {self.prepare_time} |
| Processing data | {self.data_time} |
| Generating result | {self.result_time} |
| Device | {self.device} |

## Saving model
| Variable | Value |
| --- | --- |
| Saving | {not self.dont_save_model} |'''
    
        if not self.dont_save_model:
            message += f'''
| State dictionary | {not self.save_not_as_statedict} |'''

        with open(os.path.join(self.directory_output_folder,'log.md'), 'w') as f:
            f.write(message)