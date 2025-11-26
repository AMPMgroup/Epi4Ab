import pandas as pd
import os
from tqdm import tqdm

def extract_charge(pdb_df, logging):
    for pdb_id in tqdm(pdb_df.pdbID, desc='Extract charge', unit='pdb'):
        try:
            charge_file = os.path.join(logging.directory_data, pdb_id, 'pdb2pqr', 'pdb2pqr_result.pqr')
            with open(charge_file, 'r') as f:
                lines = []
                for line in f:
                    line_dict = {}
                    line_dict['resName'] = line[19:22].strip()
                    line_dict['chainId'] = line[23].strip()
                    line_dict['resId'] = line[24:31].strip()
                    line_dict['charge'] = line[59:66].strip()
                    lines.append(line_dict)
            charge_raw_dat = pd.DataFrame(lines)
            charge_raw_dat = charge_raw_dat.astype({'charge':float,'resId':int})
            charge_res_dat = charge_raw_dat.groupby(by =['resName','chainId','resId']).aggregate('sum').reset_index().sort_values(by = ['chainId','resId'])
            charge_res_dat['pdbId'] = pdb_id
            charge_res_dat = charge_res_dat[['pdbId', 'resName','chainId','resId','charge']]
            out_folder = os.path.join(logging.directory_data, pdb_id, 'charge')
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            charge_res_dat[['pdbId', 'resName', 'chainId']] = charge_res_dat[['pdbId', 'resName', 'chainId']].astype('string')
            charge_res_dat.to_parquet(os.path.join(out_folder, 'charge_result.parquet'))
            # charge_res_dat.to_csv(os.path.join(out_folder, 'charge_result.txt'), index = None)
        except:
            logging.error_charge.append(pdb_id)

    if not logging.error_charge:
        logging.message += '''
All pdb charge have been extracted successfully.'''
    else:
        logging.message += f'''
No charge pdb(s): {logging.error_charge}'''