import argparse

def initiate_argument():
    parser = argparse.ArgumentParser(description='Training for model for protein interface prediction.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Directory
    parser.add_argument('directory_data',
                        help='path to nodes_edges folder', 
                        nargs='?')
    parser.add_argument('directory_processed_data',
                        help='path to processed_data folder', 
                        nargs='?')
    parser.add_argument('directory_pdb_list',
                        help='path to list of pdb', 
                        nargs='?')
    parser.add_argument('directory_model_folder',
                        help='model folder name',
                        nargs='?')
    parser.add_argument('directory_output',
                        help='model folder name',
                        nargs='?', default='./use_model/output')
    parser.add_argument('--use_region',
                        help='Using only region near interface',
                        action='store_true', default=False)
    parser.add_argument('--use_relaxed',
                        help='Using relax to train',
                        action='store_true')
    parser.add_argument('--plot_network',
                        help='Plot networkx for test PDBs',
                        action='store_true', default=False)
    # Seed
    parser.add_argument('--networkx_seed', metavar='',
                        help='Setting seed for networkx',
                        default=1, type=int)
    # Change Antibody feature
    parser.add_argument('--ab_feature_input',
                        help='Condition for change Ab feature in feature data',
                        action='store_true', default=False)
    
    return parser.parse_args()