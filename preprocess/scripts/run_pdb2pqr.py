import os
from subprocess import run
from tqdm import tqdm

def run_pdb2pqr(pdb_df, logging):
    for pdbId in tqdm(pdb_df.pdbID, desc = 'Run PDB2PQR', unit='pdb'):
        pdb_id_path = os.path.join(logging.directory_data, pdbId)
        pdb2pqr_out = os.path.join(pdb_id_path, 'pdb2pqr')
        pdb_file = os.path.join(pdb_id_path, 'lig.pdb')
        if not os.path.exists(pdb2pqr_out):
            os.mkdir(pdb2pqr_out)
        result_file = os.path.join(pdb2pqr_out, 'pdb2pqr_result.pqr')
        if os.path.exists(result_file):
            os.remove(result_file) 
        cmd = ['pdb2pqr', pdb_file, result_file, '--ff=AMBER', '--keep-chain', '--whitespace']
        run(cmd)
        if not os.path.exists(result_file):
            logging.error_pdb2pqr.append(pdbId)
    
    if not logging.error_pdb2pqr:
        logging.message += '''
All the pdbs have been run pdb2pqr successfully.'''
    else:
        logging.message += f'''
Failed pdb2pqr pdb(s): {logging.error_pdb2pqr}'''