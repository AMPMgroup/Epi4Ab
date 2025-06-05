import argparse

def initiate_argument():
    parser = argparse.ArgumentParser(description='Training for model for protein interface prediction.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Directory
    parser.add_argument('saved_params',
                        help='Saving used parameters',
                        nargs='?')
    parser.add_argument('directory_data',
                        help='path to nodes_edges folder', 
                        nargs='?')
    parser.add_argument('directory_processed_data',
                        help='path to processed_data folder', 
                        nargs='?')
    parser.add_argument('directory_pdb_list',
                        help='path to list of pdb', 
                        nargs='?')
    parser.add_argument('directory_unseen_pdb_list',
                        help='path to list of unseen pdb', 
                        nargs='?')
    parser.add_argument('directory_af_pdb_list',
                        help='path to list of alpha fold pdb', 
                        nargs='?')
    parser.add_argument('directory_unseen_af_pdb_list',
                        help='path to list of unseen alpha fold pdb', 
                        nargs='?')
    parser.add_argument('directory_output',
                        help='path to output folder', 
                        nargs='?', default='./output')
    # parser.add_argument('directory_mafft',
    #                     help='location of mafft', 
    #                     nargs='?', default='./home')
    # Random
    parser.add_argument('--torch_seed', metavar='',
                        help='Setting seed for torch',
                        default=1, type=int)
    parser.add_argument('--sklearn_seed', metavar='',
                        help='Setting seed for sklearn',
                        default=1, type=int)
    parser.add_argument('--networkx_seed', metavar='',
                        help='Setting seed for networkx',
                        default=1, type=int)
    # Data
    parser.add_argument('--edge_type',
                        help='Choose network type for edge',
                        default='dist', type=str, choices=['dist','comm','distComm'])
    parser.add_argument('--edge_threshold', metavar='',
                        help='Maximum distance for 2 nodes to have connection. Dataset max is 10A',
                        default=10, type=float)
    parser.add_argument('--feature_version',
                        help='Choose version of features',
                        default='v1.0.1', choices=['v1.0.0', 'v1.0.1', 'v1.0.2'])
    parser.add_argument('--use_region',
                        help='Using only region near interface',
                        action='store_true')
    parser.add_argument('--use_relaxed',
                        help='Using relax to train',
                        action='store_true')
    parser.add_argument('--use_alphafold',
                        help='Using alphafold to train and test',
                        action='store_true')
    parser.add_argument('--num_kfold', metavar='',
                        help='Number of KFOLD',
                        default=3, type=int)
    parser.add_argument('--out_label', metavar='',
                        help='Number of output label',
                        default=3, type=int)
    # parser.add_argument('--use_reciprocal',
    #                     help='Using order 0 to 4, for 0 is interface',
    #                     action='store_true', default=True)
    parser.add_argument('--bias_distance', metavar='',
                        help='Value for substracting distance of attribute',
                        default=0, type=int)
    parser.add_argument('--softmax_data',
                        help='Set softmax layer for data before training',
                        action='store_true')
    parser.add_argument('--not_normalize_data',
                        help='No normalizing data before training',
                        action='store_true')
    # Label
    parser.add_argument('--label_from_ellipro_pi',
                        help='Label for training from ElliPro result. True for PI, False for linear',
                        action='store_true')
    # Token
    parser.add_argument('--use_token',
                        help='Use token representing vh vl family',
                        action='store_true')
    parser.add_argument('--token_dim', metavar='',
                        help='Dimension size of token',
                        default=3, type=int)
    # Pre-trained
    parser.add_argument('--use_pretrained',
                        help='Using pre-trained',
                        action='store_true')
    parser.add_argument('--pretrained_model',
                        help='Pre-trained model name',
                        default='protBERT', type=str, choices=['protBERT','ESM2_t6','ESM2_t12','ESM2_t30','ESM2_t33','ESM2_t36'])
    parser.add_argument('--freeze_pretrained',
                        help='While using pre-trained, the pre-trained model can not be updated.',
                        action='store_false')
    parser.add_argument('--use_seq_ff',
                        help='Modify the dimension of pretrained by adding feeding forward',
                        action='store_true')
    parser.add_argument('--seq_ff_dim',
                        help='Dimension of 1st Linear in antiberty feed forward',
                        default=128, type=int)
    parser.add_argument('--seq_ff_out',
                        help='Dimension of 2nd Linear in antiberty feed forward',
                        default=16, type=int)
    parser.add_argument('--seq_ff_dropout',
                        help='Dropout in antiberty feed forward',
                        default=0.2, type=float)
    # Struct of version 1.0.1
    parser.add_argument('--use_struct',
                        help='Using struct data',
                        action='store_true')
    parser.add_argument('--use_deep_shallow',
                        help='Using deep and shallow in training based on resDepth',
                        action='store_true')
    parser.add_argument('--shallow_layer',
                        help='Set starting layer for shallow training, layer index start as 0',
                        default=0, type=int)
    parser.add_argument('--shallow_cutoff',
                        help='Set cutoff for resDepth',
                        default=2.0, type=float)
    # AntiBERTy
    parser.add_argument('--use_mha',
                        help='Using Multihead Attention for antiberty',
                        action='store_true')
    parser.add_argument('--mha_head',
                        help='Number of head for MultiheadAttention',
                        default=2, type=int)
    parser.add_argument('--max_antigen_len',
                        help='Max length of antigen',
                        default=700, type=int)
    parser.add_argument('--use_antiberty',
                        help='Using AntiBERTy for H3',
                        action='store_true')
    parser.add_argument('--antiberty_max_len',
                        help='Max length of H3 used for antiberty',
                        default=30, type=int)
    parser.add_argument('--antiberty_ff_dim',
                        help='Dimension of 1st Linear in antiberty feed forward',
                        default=128, type=int)
    parser.add_argument('--antiberty_ff_out',
                        help='Dimension of 2nd Linear in antiberty feed forward',
                        default=16, type=int)
    parser.add_argument('--antiberty_ff_dropout',
                        help='Dropout in antiberty feed forward',
                        default=0.2, type=float)
    # Graph Model
    parser.add_argument('--use_base_model',
                        help='Using Conv in layers, also with fc_out',
                        action='store_true')
    parser.add_argument('--not_include_gat',
                        help='Using Cheb as fc_out layer rather than GAT in transformer model',
                        action='store_true')
    parser.add_argument('--softmax_output',
                        help='Using softmax after model output',
                        action='store_true')
    parser.add_argument('--model_name', metavar='',
                        help='Select model. Please see the model script for selection',
                        default='GNNNaive', type=str, choices=['GNNNaive', 'GNNResNet', 'EpiObject','EpiRegion','EncoderGraph'])
    parser.add_argument('--model_block', metavar='',
                        help='Select block for multilayer model. Please see the model script for selection. If use_base_model, block is "Base"',
                        default='GNNNaiveBlock_Cheb', type=str, choices=['GNNNaiveBlock_Cheb', 'GNNNaiveBlock_ChebGat',
                                                                         'GNNResNetBlock_Cheb','GNNResNetBlock_ChebGat'])
    parser.add_argument('--model_description', 
                        help='Description for the model', 
                        nargs='?', default=None)
    parser.add_argument('--hidden_channel', metavar='',
                        help='Number of hidden channel, could input with multiple values',
                        default=[32], type=int, nargs='*')
    parser.add_argument('--filter_size', metavar='',
                        help='Number of k for ChebConv',
                        default=[6], type=int, nargs='*')
    parser.add_argument('--attention_head', metavar='',
                        help='Number of attention head for GAT',
                        default=[4], type=int, nargs='*')
    parser.add_argument('--initial_process_pretrained_weight',
                        help='Weight of pre-trained data in initial_process',
                        default=1, type=int)
    parser.add_argument('--initial_process_struct_weight',
                        help='Weight of struct data in initial_process',
                        default=1, type=int)
    parser.add_argument('--initial_process_antiberty_weight',
                        help='Weight of antiberty data in initial_process',
                        default=1, type=int)
    parser.add_argument('--initial_process_token_weight',
                        help='Weight of token data in initial_process',
                        default=1, type=int)
    parser.add_argument('--gat_concat',
                        help='Using concat for feature in GATConv within block',
                        action='store_true')
    parser.add_argument('--block_norm',
                        help='Using batch norm within block',
                        default=None, type=str, choices=[None, 'batchnorm', 'graphnorm'])
    parser.add_argument('--block_norm_eps',
                        help='A value added to the denominator for numerical stability in batch norm',
                        default=1e-5, type=float)
    parser.add_argument('--block_norm_momentum',
                        help='The value used for the running mean and running variance computation in batch norm',
                        default=0.1, type=float)
    # Loss function
    parser.add_argument('--loss_function', metavar='',
                        help='Select loss function',
                        default='cross_entropy', type=str, choices=['cross_entropy','mse'])
    parser.add_argument('--cross_entropy_weight', metavar='',
                        help='Add weight for cross entropy',
                        default=None, type=float, nargs=3)
    parser.add_argument('--mse_threshold', metavar='',
                        help='Number of power of attribute',
                        default=0.75, type=float)
    # Dropout
    parser.add_argument('--drop_out', metavar='',
                        help='Drop out percentage in model architecture, could input with multiple values',
                        default=[0.5], type=float, nargs='*')
    parser.add_argument('--num_layers',
                        help='Number of layers in GNN model',
                        default=1, type=int)
    parser.add_argument('--dropout_edge_p',
                        help='Dropout probability for dropout edge',
                        default=None, type=float)
    # Optimizer
    parser.add_argument('--optimizer_method', metavar='',
                        help='Select optimizer method',
                        default='adam', type=str, choices=['adam','momentum','sgd'])
    parser.add_argument('--learning_rate', metavar='',
                        help='Learning rate for optimizer',
                        default=0.001, type=float)
    parser.add_argument('--weight_decay', metavar='',
                        help='Weight decay for optimizer',
                        default=5e-4, type=float)
    parser.add_argument('--momentum', metavar='',
                        help='Momentum for optimizer',
                        default=0.9, type=float)
    # Train
    parser.add_argument('--epoch_number', metavar='',
                        help='Number of epoches',
                        default=200, type=int)
    parser.add_argument('--batch_size', metavar='',
                        help='Size of batch',
                        default=1, type=int)
    # Train attribute
    parser.add_argument('--gradient_attribute',
                        help='Adding gradient for attribute',
                        action='store_true')
    parser.add_argument('--gradient_attribute_with_bias',
                        help='Adding bias to gradient for attribute',
                        action='store_true')
    parser.add_argument('--attribute_no_bond',
                        help='NOT using bond potential in attribute',
                        action='store_true')
    parser.add_argument('--attribute_no_lj',
                        help='NOT using Lennard-Jones potential in attribute',
                        action='store_true')
    parser.add_argument('--attribute_no_charge',
                        help='NOT using charge potential in attribute',
                        action='store_true')   
    parser.add_argument('--attribute_weight', metavar='',
                        help='Weight of edge attribute',
                        default=[1.0,1.0,1.0], type=float, nargs=3)
    # Run type
    parser.add_argument('--train_all', metavar='',
                        help='Run data all data for unseen',
                        default='no', type=str, choices=['yes','with_validation','no'])
    # Save model
    parser.add_argument('--dont_save_model',
                        help='Not saving model when train_all',
                        action='store_true')
    parser.add_argument('--save_not_as_statedict',
                        help='Save model NOT as state dictionary',
                        action='store_true')
    # Save networkx
    parser.add_argument('--plot_network',
                        help='Plot networkx for test PDBs',
                        action='store_true')

    return parser.parse_args()