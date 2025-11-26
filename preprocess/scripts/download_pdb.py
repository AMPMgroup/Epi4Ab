import os
from tqdm import tqdm
from Bio.PDB.PDBList import PDBList

def download_pdb_file(pdb_series, pdb_dir, pdb_down, logging):
    pdb = pdb_series.pdb.item()
    pdb_down.retrieve_pdb_file(pdb, pdir = pdb_dir, file_format = 'mmCif')

    if not os.path.exists(os.path.join(pdb_dir, f'{pdb}.cif')):
        logging.error_download.append(pdb)
    
def download_pdb(pdb_df, logging):
    pdb_down = PDBList(verbose = False)
    for pdb_id in tqdm(pdb_df.pdbID, desc='Download', unit='pdb'):
        pdb_id_info = pdb_df[pdb_df.pdbID == pdb_id]
        pdb_id_dir = os.path.join(logging.directory_data, pdb_id)
        if not os.path.exists(pdb_id_dir):
            os.mkdir(pdb_id_dir)
        try:
            download_pdb_file(pdb_id_info, pdb_id_dir, pdb_down, logging)
        except:
            print(pdb_id)
            logging.error_download.append(pdb_id)

    if not logging.error_download:
        logging.message += '''
All the pdbs have been downloaded successfully.'''
    else:
        logging.message += f'''
Not downloaded pdb(s): {logging.error_download}'''

def create_folder(pdb_df, logging):
    for pdb_id in tqdm(pdb_df.pdbID, desc='Create folder', unit='pdb'):
        pdb_id_dir = os.path.join(logging.directory_data, pdb_id)
        if not os.path.exists(pdb_id_dir):
            os.mkdir(pdb_id_dir)

