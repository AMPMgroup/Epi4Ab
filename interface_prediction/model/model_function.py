from .model_class.graph import *
from .model_class.initial_process import InitialProcess
from .normalizer import Normalizer

def choose_model(logging):
    initial_process = InitialProcess(logging.combine_input,
                                    logging.use_pretrained,
                                    logging.pretrained_model,
                                    logging.freeze_pretrained,
                                    logging.use_token,
                                    logging.token_size,
                                    logging.token_dim,
                                    logging.use_continuous,
                                    logging.reserved_columns,
                                    logging.continuous_embed_dim,
                                    logging.use_struct,
                                    logging.initial_process_weight_dict,
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
                        logging.attention_head, 
                        logging.filter_size,
                        logging.gradient_attribute,
                        logging.gradient_attribute_with_bias,
                        logging.model_block,
                        logging.use_base_model,
                        normalizer)
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
                        normalizer)
    
    else:
        raise TypeError(f'Model {logging.model_name} is not defined.')