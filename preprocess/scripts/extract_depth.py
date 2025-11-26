import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import ResidueDepth
from tqdm import tqdm

def get_depth_df(rd, pdb_id, chain_id):
    residue_name = []
    residue_id = []
    residue_depth = []
    ca_depth = []
    for res_rd in rd:
        residue_name.append(res_rd[0].get_resname())
        residue_id.append(res_rd[0].get_id()[1])
        residue_depth.append(res_rd[1][0])
        ca_depth.append(res_rd[1][1])
    return pd.DataFrame({'pdbId':pdb_id,
                            'resName':residue_name,
                            'chainId':chain_id,
                            'resId': residue_id,
                            'resDepth':residue_depth,
                            'caDepth':ca_depth})

def extract_depth(pdb_df, logging):
    parser = PDBParser(QUIET=True)
    for pdb_id in tqdm(pdb_df.pdbID, desc='Extract depth', unit='pdb'):  
        pdb_id_path = os.path.join(logging.directory_data, pdb_id)
        output_path = os.path.join(pdb_id_path, 'depth')
        lig_file_path = os.path.join(pdb_id_path, 'lig.pdb')
        structure = parser.get_structure(pdb_id, lig_file_path)
        for chain in structure[0]:
            rd = ResidueDepth(chain)
            lig_df = get_depth_df(rd, pdb_id, chain.id)

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        lig_df.to_parquet(os.path.join(output_path,'depth_result.parquet'))
        # except:
        #     logging.error_depth.append(pdb_id)
    if not logging.error_depth:
        logging.message += '''
All pdb depth have been extracted successfully.'''
    else:
        logging.message += f'''
No depth pdb(s): {logging.error_depth}'''
        print(logging.error_depth)    