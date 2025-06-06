import torch
from datetime import date
import os
import json
from .set_up import multilayer_warning

class ModelLogging:
    def __init__(self, args):
        self.code_version = '2024-12-30'
        # random
        self.torch_seed = args.torch_seed
        self.sklearn_seed = args.sklearn_seed
        self.networkx_seed = args.networkx_seed
        # file
        self.feature_file = None
        self.edge_attribute_file = None
        self.edge_attribute_charge_file = None
        self.edge_index_file = None
        self.node_label_file = None
        # data
        self.feature_version = args.feature_version
        self.continuous_columns = None
        self.ag_continuous_columns = None
        self.ab_continuous_columns = None
        self.onehot_columns = None
        self.ag_onehot_columns = None
        self.ab_onehot_columns = None
        self.amino_acid_list = None
        self.feature_columns_number = 0
        self.edge_type = args.edge_type
        assert (args.edge_threshold >= 3) & (args.edge_threshold <= 10), f'Threshold value {args.edge_threshold} should be between 3 and 10.'
        self.edge_threshold = args.edge_threshold
        self.use_region = args.use_region
        self.use_relaxed = args.use_relaxed
        self.use_alphafold = args.use_alphafold
        self.num_kfold = args.num_kfold
        self.softmax_data = args.softmax_data
        self.not_normalize_data = args.not_normalize_data
        # self.use_reciprocal = args.use_reciprocal
        self.bias_distance = args.bias_distance
        self.mse_threshold = args.mse_threshold
        self.ab_feature_input:bool = False
        self.error_data_process_list = []
        # Label
        self.label_from_ellipro_pi = args.label_from_ellipro_pi
        # pre-trained
        self.use_pretrained = True if args.feature_version == 'v1.0.1' else args.use_pretrained
        self.pretrained_model = args.pretrained_model if self.use_pretrained else None
        if self.pretrained_model == 'protBERT':
            self.pretrained_dim = 1024
        elif self.pretrained_model == 'ESM2_t6':
            self.pretrained_dim = 320
        elif self.pretrained_model == 'ESM2_t12':
            self.pretrained_dim = 480
        elif self.pretrained_model == 'ESM2_t30':
            self.pretrained_dim = 640
        elif self.pretrained_model == 'ESM2_t33':
            self.pretrained_dim = 1280
        elif self.pretrained_model == 'ESM2_t36':
            self.pretrained_dim = 2560
        else:
            self.pretrained_dim = None
        self.freeze_pretrained = args.freeze_pretrained if self.use_pretrained else None
        self.use_seq_ff = args.use_seq_ff
        self.seq_ff_dim = args.seq_ff_dim
        self.seq_ff_out = args.seq_ff_out
        self.seq_ff_dropout = args.seq_ff_dropout
        # Continuous
        self.continuous_embed_dim = None
        self.reserved_columns = 0
        # Token
        self.use_token = args.use_token
        self.token_size = 0
        self.token_dim = args.token_dim
        # AntiBERTy
        self.use_antiberty = args.use_antiberty
        self.antiberty_max_len = args.antiberty_max_len
        self.antiberty_ff_dim = args.antiberty_ff_dim
        self.antiberty_ff_out = args.antiberty_ff_out
        self.antiberty_ff_dropout = args.antiberty_ff_dropout
        self.antiberty_dim = 512
        self.use_mha = args.use_mha
        self.mha_head = args.mha_head
        self.max_antigen_len = args.max_antigen_len
        self.mha_num_layers = args.mha_num_layers
        # Struct
        self.use_struct = args.use_struct
        self.use_deep_shallow = args.use_deep_shallow
        self.shallow_layer = args.shallow_layer
        assert self.shallow_layer >= 0, f'Shallow layer {self.shallow_layer} should not be negative'
        self.shallow_cutoff = args.shallow_cutoff
        assert self.shallow_cutoff >= 0, f'Shallow cutoff {self.shallow_cutoff} should not be negative'
        self.resDepth_index = None
        # graph model
        self.use_base_model = args.use_base_model
        self.not_include_gat = args.not_include_gat
        self.softmax_output = args.softmax_output
        self.model_name = args.model_name
        self.object_model = True if self.model_name in ['EpiObject','EpiRegion'] else False
        self.model_block = args.model_block
        self.model_description = args.model_description
        self.model_architecture = None
        self.in_feature = 0
        self.hidden_channel = args.hidden_channel
        if args.loss_function == 'mse':
            self.out_label = 1
        else:
            self.out_label = args.out_label
        self.filter_size = args.filter_size
        self.attention_head = args.attention_head
        self.drop_out = args.drop_out
        self.num_layers = args.num_layers
        self.dropout_edge_p = args.dropout_edge_p
        multilayer_warning(self.num_layers, self.hidden_channel, 'hidden channel')
        multilayer_warning(self.num_layers, self.filter_size, 'filter size')
        multilayer_warning(self.num_layers, self.drop_out, 'drop out')
        self.initial_process_weight_dict = {
            'pre-trained' :1 if args.feature_version in ['v1.0.0','v1.0.1'] else args.initial_process_pretrained_weight,
            'struct': 1 if args.feature_version in ['v1.0.0','v1.0.1'] else args.initial_process_struct_weight,
            'antiberty': 1 if args.feature_version in ['v1.0.0','v1.0.1'] else args.initial_process_antiberty_weight,
            'token': 1 if args.feature_version in ['v1.0.0','v1.0.1'] else args.initial_process_token_weight
        }
        self.gat_concat = args.gat_concat
        self.block_norm = args.block_norm
        self.block_norm_eps = args.block_norm_eps
        self.block_norm_momentum = args.block_norm_momentum
        # Loss function
        self.loss_function = args.loss_function
        if args.cross_entropy_weight is not None:
            assert len(args.cross_entropy_weight) == args.out_label, f'Number of Cross entropy weight ({args.cross_entropy_weight}) and number of out label ({args.out_label}) should be equal.'
        self.cross_entropy_weight = args.cross_entropy_weight
        # Optimizer
        self.optimizer_method = args.optimizer_method
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        # Train
        self.epoch_number = args.epoch_number
        self.batch_size = args.batch_size
        self.train_all = args.train_all
        # Train attribute
        self.gradient_attribute = args.gradient_attribute
        self.gradient_attribute_with_bias = args.gradient_attribute_with_bias
        self.attribute_no_bond = args.attribute_no_bond
        self.attribute_no_lj = args.attribute_no_lj
        self.attribute_no_charge = args.attribute_no_charge
        self.attribute_weight = args.attribute_weight
        self.gradient_attribute_weight = []
        self.gradient_attribute_bias = []
        self.info_record_columns = None
        # Metric
        if self.object_model:
            self.evaluation_columns = ["Recall","Precision","f1", "Average Precision"]
        elif self.model_name == 'ParaSequence':
            self.evaluation_columns = ["Global Score Percent", "Local Score Percent", "Cosine Similarity", "Pairwise Score Percent"]
        else:
            self.evaluation_columns = ["Recall","Precision","f1","Accuracy","ROC AUC","Average Precision"]
        # GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # directory
        self.run_date = date.today()
        self.directory_data:str = args.directory_data
        self.directory_processed_data:str = args.directory_processed_data
        self.directory_pdb_list:str = args.directory_pdb_list
        self.directory_unseen_pdb_list:str = args.directory_unseen_pdb_list
        self.directory_af_pdb_list:str = args.directory_af_pdb_list
        self.directory_unseen_af_pdb_list:str = args.directory_unseen_af_pdb_list
        self.directory_output:str = args.directory_output
        self.directory_output_folder:str = None
        self.directory_test_record:str = None
        # self.directory_mafft:str = None
        # Save model
        self.dont_save_model = args.dont_save_model
        self.save_not_as_statedict = args.save_not_as_statedict
        # Plot networkx
        self.plot_network = args.plot_network
        # Time
        self.data_time = None
        self.train_time = None
        self.prepare_time = None
        self.result_time = None

    def __iter__(self):
        for attr, value in self.__dict__.items():
            if attr in ['device', 'run_date', 'data_time', 'train_time', 'prepare_time', 'result_time']:
                yield attr, str(value)
            else:
                yield attr, value

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
| Use MHA | {self.use_mha} |
| MHA head | {self.mha_head} |
| MHA number of layers | {self.mha_num_layers} |
| Max antigen length | {self.max_antigen_len} |
| Use AntiBERTy | {self.use_antiberty} |
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
| Methods | {self.evaluation_columns} |

## Directory
| Variable | Value |
| --- | --- |
| Data | {self.directory_data} |
| Processed data | {self.directory_processed_data} |
| PDB list | {self.directory_pdb_list} |
| PDB unseen list | {self.directory_unseen_pdb_list} |
| AF PDB list | {self.directory_af_pdb_list} |
| AF PDB unseen list | {self.directory_unseen_af_pdb_list} |
| Output | {self.directory_output} |

## Runtime
| Variable | Value |
| --- | --- |
| Preparing | {self.prepare_time} |
| Processing data | {self.data_time} |
| Training | {self.train_time} |
| Generating result | {self.result_time} |
| Device | {self.device} |

## Saving model
| Variable | Value |
| --- | --- |
| Saving | {not self.dont_save_model} |'''
    
        if not self.dont_save_model:
            message += f'''
| State dictionary | {not self.save_not_as_statedict} |'''
            
            with open(os.path.join(self.directory_output_folder, 'log.json'), 'w') as file:
                json.dump(dict(self), file)

        with open(os.path.join(self.directory_output_folder,'log.md'), 'w') as f:
            f.write(message)

        