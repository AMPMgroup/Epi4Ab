import argparse

def initiate_argument():
    parser = argparse.ArgumentParser(description='Training for model for protein interface prediction.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Directory
    parser.add_argument('directory_metadata',
                        help='path to metadata file', 
                        nargs='?')
    parser.add_argument('directory_output',
                        help='path to download pdb folder', 
                        nargs='?')
    parser.add_argument('--download_pdb',
                        help='flag to download pdb files',
                        action='store_true')

    return parser.parse_args()