import pandas as pd
import os
import argparse
from datetime import date
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Training for model for protein interface prediction.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Directory
parser.add_argument('directory_metadata',
                    help='path to metadata file')
parser.add_argument('directory_output',
                    help='path to download pdb folder')

args = parser.parse_args()
run_date = date.today()
meta_df = pd.read_csv(args.directory_metadata)

all_profile = pd.DataFrame()
all_linear_profile = pd.DataFrame()
all_sequence = {}
error_list = []

for pdb_id in tqdm(meta_df.pdbID, desc='Combine profile', unit='pdb'):
    # try:
    pdb_path = os.path.join(args.directory_output, 'data', pdb_id)
    # PI
    profile_file = os.path.join(pdb_path, 'pdb_profile.parquet')
    pdb_profile = pd.read_parquet(profile_file)
    if all_profile.empty:
        all_profile = pdb_profile.copy()
    else:
        all_profile = pd.concat([all_profile, pdb_profile])

all_profile[['pdbId','chainId','resName','resShort','chainType']] = all_profile[['pdbId','chainId','resName','resShort','chainType']].astype('string')
all_profile[['H3_score','L1_score']] = all_profile[['H3_score','L1_score']].astype(float)
all_profile = all_profile.sort_values(by=['pdbId','chainId','resId'])
all_profile.to_csv(os.path.join(args.directory_output, f'combined_profile.txt'), index=None)

if error_list != []:
    print(f'Error pdb(s): {error_list}')
else:
    print('All profile combined successfully.')