import os
from tqdm import tqdm
import json
import pandas as pd

def extract_chain_sequence(chain_df, output_folder, chain_type):
    seq_str = ' '.join([str(res) for res in chain_df.resShort])
    # interface_lst = []
    # for res in chain_df.isInterface:
    #     if res == 0:
    #         interface_lst.append('B')
    #     else:
    #         interface_lst.append('C')
    # interface_str = ' '.join(interface_lst)
    chain_dict = {
        'pdb_sequence':seq_str
        # 'interface_sequence':interface_str
    }
    with open(os.path.join(output_folder, f'{chain_type}_sequence.json'), 'w') as f:
            json.dump(chain_dict, f)
    return seq_str

def extract_cdr_sequence(pdb_series, output_folder):
    chain_dict = {}
    cdr_name = ['H1_seq','H2_seq','H3_seq','L1_seq','L2_seq','L3_seq']

    chain_dict = {cdr:pdb_series[cdr].values[0] for cdr in cdr_name}
    with open(os.path.join(output_folder, f'cdr_sequence.json'), 'w') as f:
        json.dump(chain_dict, f)
    

def extract_sequence(pdb_df, logging):
    for pdb_id in tqdm(pdb_df.pdbID, desc = 'Extract sequence', unit='pdb'):
        # Antigen
        data_path = os.path.join(logging.directory_data, pdb_id)
        ag_profile_file = pd.read_parquet(os.path.join(data_path,'pdb_profile.parquet'))

        ag_profile = ag_profile_file[ag_profile_file.chainType == 'antigen']
        
        output_folder = os.path.join(data_path, 'sequence')
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        _ = extract_chain_sequence(ag_profile, output_folder, 'antigen')
        # Antibody
        filtered_df = pdb_df[pdb_df.pdbID == pdb_id]
        extract_cdr_sequence(filtered_df, output_folder)
