import pandas as pd
import os
import argparse
from tqdm import tqdm
from pathlib import Path
import time
from scripts.alignment_CDRs import mafft_MSA

parser = argparse.ArgumentParser(description='Training for model for protein interface prediction.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Directory
parser.add_argument('directory_pdb_info',
                    help='path to user input file')
parser.add_argument('directory_input_data',
                    help='path to input data folder')
parser.add_argument('mafft_path',
                    help='path to mafft executable')

args = parser.parse_args()

pdb_info_df = pd.read_csv(args.directory_pdb_info)
df_model = pd.read_csv(Path(__file__).parent / 'scripts/pdb4ML_all_notRelax_VHVLh3l1_profile.csv')

pdb_info_df.loc[:,'H1_len'] = pdb_info_df.H1_seq.str.len()
pdb_info_df.loc[:,'H2_len'] = pdb_info_df.H2_seq.str.len()
pdb_info_df.loc[:,'H3_len'] = pdb_info_df.H3_seq.str.len()
pdb_info_df.loc[:,'L1_len'] = pdb_info_df.L1_seq.str.len()
pdb_info_df.loc[:,'L2_len'] = pdb_info_df.L2_seq.str.len()
pdb_info_df.loc[:,'L3_len'] = pdb_info_df.L3_seq.str.len()

unseen_DICT = {'pdbID':[],'H3_score':[],'L1_score':[],'VHVLh3l1':[],'H3_template':[],'L1_template':[],'align_status':[]}

cnt=0
for pdb in tqdm(pdb_info_df['pdbID'].values, desc='Process alignment'):
    #print('\n',pdb)
    VH = pdb_info_df.loc[pdb_info_df['pdbID']==pdb, 'VH_fam'].values[0]
    VL = pdb_info_df.loc[pdb_info_df['pdbID']==pdb, 'VL_fam'].values[0]
    h3_len = pdb_info_df.loc[pdb_info_df['pdbID']==pdb, 'H3_len'].values[0]
    l1_len = pdb_info_df.loc[pdb_info_df['pdbID']==pdb, 'L1_len'].values[0]
    h3_seq = pdb_info_df.loc[pdb_info_df['pdbID']==pdb, 'H3_seq'].values[0]
    l1_seq = pdb_info_df.loc[pdb_info_df['pdbID']==pdb, 'L1_seq'].values[0]
    # VH,VL,h3_len,l1_len,h3_seq,l1_seq = pdb_info_df.loc[pdb_info_df['pdbID']==pdb].values.flatten().tolist()[1:]

    unseen_DICT['pdbID'].append(pdb)
    unseen_DICT['VHVLh3l1'].append(f'{VH}_{VL}_{h3_len}_{l1_len}')


    unseen_tem = f'{VH}_{VL}_{h3_len}_{l1_len}'
    try:
        h3_template,l1_template = df_model.loc[df_model['H3_cluster'] == unseen_tem].values.flatten().tolist()[3:5]
        ## align
        _, _, H3_dict = mafft_MSA([h3_template,h3_seq],'H3', args.mafft_path)
        # print(H3_dict)
        H3_score = H3_dict[1]
        unseen_DICT['H3_score'].append(H3_score)
        unseen_DICT['H3_template'].append(h3_template)
        #print(h3_template,h3_seq,H3_score)

        _, _, L1_dict = mafft_MSA([l1_template,l1_seq],'L1', args.mafft_path)
        L1_score = L1_dict[1]
        unseen_DICT['L1_score'].append(L1_score)
        unseen_DICT['L1_template'].append(l1_template)

        unseen_DICT['align_status'].append('aligned')
        cnt+=1

    except:
        #print(f'ERROR -- {VH}_{VL} not found in model_clusters')
        unseen_DICT['H3_score'].append('nil')
        unseen_DICT['L1_score'].append('nil')
        unseen_DICT['H3_template'].append('NA')
        unseen_DICT['L1_template'].append('NA')
        unseen_DICT['align_status'].append('clusters_notExist')

# print(f'{cnt} unseen pdbs aligned -- {len(pdb_info_df) - cnt}')
os.system('rm H3_msa.in H3_msa.out L1_msa.in L1_msa.out')

df_out = pd.DataFrame(unseen_DICT)

df_out = pdb_info_df.merge(df_out[['pdbID','H3_score','L1_score','VHVLh3l1','H3_template','L1_template']],on='pdbID')
df_out.to_csv(os.path.join(args.directory_input_data,'metadata.csv'), index=None)
