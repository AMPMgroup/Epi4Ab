from .model_class.graph import *
from .model_class.initial_process import InitialProcess
from .normalizer import Normalizer

def choose_model(logging):
    initial_process = InitialProcess(logging.use_pretrained,
                                    logging.pretrained_model,
                                    logging.freeze_pretrained,
                                    logging.use_seq_ff,
                                    logging.pretrained_dim,
                                    logging.seq_ff_dim,
                                    logging.seq_ff_out,
                                    logging.seq_ff_dropout,
                                    logging.use_antiberty,
                                    logging.antiberty_dim,
                                    logging.antiberty_ff_dim,
                                    logging.antiberty_ff_out,
                                    logging.antiberty_ff_dropout,
                                    logging.antiberty_max_len,
                                    logging.use_token,
                                    len(logging.ab_onehot_vh_columns), 
                                    len(logging.ab_onehot_vl_columns),
                                    logging.token_dim,
                                    # logging.reserved_columns,
                                    # logging.continuous_embed_dim,
                                    logging.use_struct,
                                    logging.use_deep_shallow,
                                    logging.shallow_cutoff,
                                    logging.resDepth_index,
                                    logging.initial_process_weight_dict,
                                    logging.use_mha,
                                    logging.mha_head,
                                    logging.max_antigen_len,
                                    logging.in_feature,
                                    logging.mha_num_layers,
                                    logging.mha_dropout,
                                    logging.device)
    normalizer = Normalizer(logging.block_norm,
                            logging.block_norm_eps,
                            logging.block_norm_momentum)
    if logging.model_name == 'GNNNaive':
        logging.model_architecture = '''
Operators: Cheb - (Block) x num_layers - (out layer)

Activation: LeakyReLU

- 01:
    - Block: Cheb
    - out layer: GAT
- 02:
    - Block: Cheb GAT
    - out layer: Linear
'''
        return GNNNaive(initial_process,
                        logging.in_feature, 
                        logging.out_label, 
                        logging.hidden_channel,
                        logging.num_layers, 
                        logging.drop_out, 
                        logging.dropout_edge_p,
                        logging.attention_head, 
                        logging.filter_size,
                        logging.gradient_attribute,
                        logging.gradient_attribute_with_bias,
                        logging.model_block,
                        logging.use_base_model,
                        normalizer,
                        logging.use_deep_shallow,
                        logging.shallow_layer)
    elif logging.model_name == 'GNNResNet':
        logging.model_architecture = '''
Operators: Cheb - (Block) x num_layers - (out layer)

Activation: LeakyReLU

- 01:
    - Block: Cheb
    - out layer: GAT
- 02:
    - Block: Cheb GAT
    - out layer: Linear
'''
        return GNNResNet(initial_process, 
                         logging.in_feature,
                        logging.out_label, 
                        logging.hidden_channel,
                        logging.num_layers, 
                        logging.drop_out, 
                        logging.dropout_edge_p,
                        logging.attention_head, 
                        logging.filter_size,
                        logging.gradient_attribute,
                        logging.gradient_attribute_with_bias,
                        logging.model_block,
                        logging.use_base_model,
                        logging.gat_concat,
                        normalizer,
                        logging.use_deep_shallow,
                        logging.shallow_layer)
    
    else:
        raise TypeError(f'Model {logging.model_name} is not defined.')