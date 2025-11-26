import os
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm

def extract_coords(pdb_df, logging):
    parser = PDBParser(QUIET=True)
    for pdb_id in tqdm(pdb_df.pdbID, desc='Extract pdb coords', unit='pdb'):
        pdb_path = os.path.join(logging.directory_data, pdb_id)
        pdb_nodes_edges = os.path.join(logging.directory_nodes_edges, pdb_id)
        if logging.process_relaxed:
            lig_file = os.path.join(logging.directory_data, pdb_id[:-3], 'lig_re.pdb')
        else:
            lig_file = os.path.join(pdb_path, 'lig.pdb')
        
        structure = parser.get_structure(pdb_id, lig_file)
        lig_coord_list = []
        structure = parser.get_structure(pdb_id, lig_file)
        for chain in structure[0]:
            for res in chain:
                lig_coord_list.append(res['CA'].get_coord())

        lig_coord_array = np.array(lig_coord_list)
        shifted_coord_array = np.stack([lig_coord_array[...,i] - np.min(lig_coord_array[...,i]) for i in range(3)]).transpose()

        all_coord_df = pd.DataFrame(shifted_coord_array,columns=['x','y','z'])
        # all_coord_df.to_parquet(os.path.join(pdb_nodes_edges,'pdb_coord.parquet'))

        assert logging is None, 'sfd'