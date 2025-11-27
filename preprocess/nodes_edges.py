# create nodes_edges

import os, sys
from tqdm import tqdm
import pandas as pd
import subprocess
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Training for model for protein interface prediction.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Directory
parser.add_argument('directory_metadata',
                    help='path to metadata file')
parser.add_argument('directory_processed',
                    help='path to download processed data folder')
parser.add_argument('directory_nodes_edges',
                    help='path to download nodes edges folder')
parser.add_argument('pymol_path',
                    help='path to pymol executable')
parser.add_argument('Catom',
                    help='atom type for CA or CB', 
                    default='CB', choices=['CA','CB'])

args = parser.parse_args()

# ======================================================================== 
profile_path = args.directory_processed
metadata_file = args.directory_metadata

cwd = args.directory_nodes_edges
if not os.path.exists(cwd):
    os.mkdir(cwd)
pymol = args.pymol_path
# ========================================================================

def prepare_nodeEdge(args):
    meta_df = pd.read_csv(metadata_file)
    
    ferr = open('errors.txt','a')
    cnt = 0

    print(f'using {metadata_file} \n--total pdbs: {len(meta_df.pdbID.values)}')
    for pdbId in tqdm(meta_df.pdbID.values, desc=f'Prepare nodes edges for {args.Catom}'):
        df_RESprofile = pd.read_parquet(os.path.join(profile_path, pdbId, 'pdb_profile.parquet'))
        # pdbId = f'{pdbId}_re'  # for re extra pdbs
        try:
            edge_attFile = os.path.join(cwd,pdbId,f'edge_attribute_dist_{args.Catom}.txt')
            if os.path.exists(edge_attFile) and os.path.getsize(edge_attFile) > 5: 
                pass
                # continue
            else:
                os.system(f'mkdir -p {cwd}/{pdbId}')
                os.chdir(f'{cwd}/{pdbId}')
                tensor_format(pdbId, df_RESprofile)
                # print(f'NOT running properly: {pdbId}')
                cnt += 1
            # print(f'{pdbId} -- {cnt}')
        except:
            ferr.write(f'{datetime.now()} -{pdbId}\n')
            #raise
            pass
    ferr.close()

def tensor_format(pdbId, df_RESprofile):

    df_Ag = df_RESprofile[df_RESprofile.pdbId == pdbId]
    
    ## 1. node_list:
    df_nlist = df_Ag[['resId','chainId']].reset_index(drop=True)
    df_nlist['resId'] = pd.to_numeric(df_nlist['resId'])   # mix types in resId column
    # df_nlist.to_parquet('./node_list.parquet',engine='fastparquet',index=False)
  

    ## 2a. node_features:
    df_nfeature = pd.read_parquet(f'{profile_path}/{pdbId}/pdb_profile.parquet')
    df_nfeature = df_Ag.drop(['pdbId','resName','chainId','chainType'],axis=1)
    df_nfeature.to_parquet('./node_feature.parquet',engine='fastparquet',index=False)

    pdb = f'{profile_path}/{pdbId}/lig.pdb'

    def pml(pdb,s_resID,s_chainID):
        ## get resi within 10A and calculate Cb-Cb distances
        with open('tmp.pml','w') as fo:
            fo.write(f'load {pdb}, mol \n')
            fo.write(f'sele near10A, resi {s_resID} and chain {s_chainID} around 10 \n')
            comm = f'"resi %s and name {args.Catom} and chain %s" %(resi,chain)'
            fo.write(f'iterate near10A and name {args.Catom}, print("##",resn,resi,chain,cmd.distance("resi {s_resID} and name {args.Catom} and chain {s_chainID}",{comm})) \n')
            #fo.write('quit \n')
    

    df_nCharge = df_Ag[['resId','chainId','charge']].reset_index(drop=True)

    EDGES = {'source':[],'target':[],'dist':[],'qi*qj':[]}
    resList_Ag = df_nlist.values.tolist()

    fo_dist = open(f'dist10A-{args.Catom}_{pdbId}.csv','w')
    for (s_resID,s_chainID) in [(i,j) for i,j in resList_Ag]:
        pml(pdb,s_resID,s_chainID)
        os.system(f'{pymol} -c tmp.pml > res10A.txt')

        source_index = df_nlist.index[df_nlist.resId == s_resID].tolist()[0]
        with open('res10A.txt') as lines:
            for line in lines:
                if '##' in line[:2]:
                    [t_resn,t_resid,t_chain,dist] = line[3:].split()

                    if 0.01 < float(dist) <= 10.0:
                        dist = '%.2f' %(float(dist))
                        # try:
                        target_index = df_nlist.index[(df_nlist.resId == int(t_resid)) & (df_nlist.chainId == t_chain)].tolist()[0]
            
                        s_charge = df_nCharge.at[source_index,'charge']
                        t_charge = df_nCharge.at[target_index,'charge'] 
                        qiqj = s_charge * t_charge

                        EDGES['source'].append(source_index)
                        EDGES['target'].append(target_index)
                        EDGES['dist'].append(dist)
                        EDGES['qi*qj'].append(qiqj)
                        
                        fo_dist.write(f'{s_resID}(id:{source_index})--{t_resid}(id:{target_index}) --dist:{dist} --charge:{qiqj}\n')

                        # except:
                        #     #raise
                        #     pass
    fo_dist.close()
    df_edge = pd.DataFrame(EDGES)
    df_eIndex = df_edge.drop(['dist','qi*qj'],axis=1)
    df_eAttribute = df_edge[['dist']]
    df_eCharge = df_edge[['qi*qj']]

    # os.system('rm res10A.txt tmp.pml')

    # write to csv
    df_eAttribute.to_csv(f'./edge_attribute_dist_{args.Catom}.txt',index=False)  

    # write to parquet

    df_eIndex.to_parquet(f'./edge_index_{args.Catom}.parquet',engine='fastparquet',index=False)  
    df_eAttribute.to_parquet(f'./edge_attribute_dist_{args.Catom}.parquet',engine='fastparquet',index=False)  
    df_eCharge.to_parquet(f'./edge_attribute_charge_{args.Catom}.parquet',engine='fastparquet',index=False) 

                            
    # '''
# ===================
prepare_nodeEdge(args)


