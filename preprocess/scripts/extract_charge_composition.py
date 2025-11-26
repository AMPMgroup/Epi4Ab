import os
from tqdm import tqdm
import json
from Bio.PDB import PDBParser
import pandas as pd
import numpy as np
from pathlib import Path

def extract_cc_from_chain(chain, aa_profile):    
    charge_types = ['negative','positive','polar','hydrophobic']
    aa_dict = {
        "ALA":0,
        "VAL":0,
        "LEU":0,
        "ILE":0,
        "PRO":0,
        "MET":0,
        "PHE":0,
        "TRP":0,
        "GLY":0,
        "SER":0,
        "THR":0,
        "CYS":0,
        "ASN":0,
        "GLN":0,
        "TYR":0,
        "LYS":0,
        "ARG":0,
        "HIS":0,
        "ASP":0,
        "GLU":0
    }
    seq_len = 0
    for res in chain:
        aa_dict[res.get_resname()] += 1
        seq_len += 1
    all_charge_arr = np.zeros(len(aa_dict))
    aa_arr = np.array(list(aa_dict.values()))
    for charge_type in charge_types:
        charge_arr = aa_arr * np.array(list(aa_profile[charge_type].values()))
        charge_arr = charge_arr/sum(charge_arr)
        assert not np.isnan(charge_arr).any(), f'''{aa_dict} 
{charge_arr}'''
        all_charge_arr += charge_arr
    for ind, key in enumerate(aa_dict.keys()):
        aa_dict[key] = all_charge_arr[ind]
    return aa_dict

def extract_cc(pdb_df, logging):
    parser = PDBParser(QUIET = True)
    
    with open(Path(__file__).parent / 'amino_acid_profile.json', 'r') as f:
        aa_profile = json.load(f)
    for pdb_id in tqdm(pdb_df.pdbID, desc='Extract Charge Composition', unit='pdb'):
        
        try:
            data_path = os.path.join(logging.directory_data, pdb_id)
            pdb_file = os.path.join(data_path, 'lig.pdb')
            structure = parser.get_structure(pdb_id, pdb_file)

            cc_profile = {}
            for chain in structure[0]:
                cc_profile[chain.id] = extract_cc_from_chain(chain, aa_profile)
                

            df = pd.DataFrame(cc_profile).reset_index(names='resName').melt(id_vars=['resName'], 
                                                                            value_vars=cc_profile.keys(), 
                                                                            var_name= 'chainId', 
                                                                            value_name='cc')
        
            output_path = os.path.join(data_path, 'charge_composition')
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            df[['resName','chainId']] = df[['resName','chainId']].astype('string')
            df.to_parquet(os.path.join(output_path, 'cc_result.parquet'))
        except:
            logging.error_charge_compostion.append(pdb_id)
    if not logging.error_charge_compostion:
        logging.message += '''
All pdb charge composition have been extracted successfully.'''
    else:
        logging.message += f'''
No charge pdb(s): {logging.error_charge_compostion}'''
