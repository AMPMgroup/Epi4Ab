import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Training for model for protein interface prediction.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Directory
parser.add_argument('directory_metadata',
                    help='path to metadata file')
parser.add_argument('directory_nodes_edges',
                    help='path to download nodes edges folder')

args = parser.parse_args()

# os_path = "/media/chinh/4tb/data"   
# data_3a_folder = f'{os_path}/Synology_AMPM/Collab_Nhan/data/'
cb_edge_index_name = 'edge_index_CB.parquet'
cb_edge_attribute_dist_name = 'edge_attribute_dist_CB.parquet'
cb_edge_attribute_charge_name = 'edge_attribute_charge_CB.parquet'

ca_edge_index_name = 'edge_index_CA.parquet'
ca_edge_attribute_dist_name = 'edge_attribute_dist_CA.parquet'
ca_edge_attribute_charge_name = 'edge_attribute_charge_CA.parquet'

node_feature_name = 'node_feature.parquet'
edge_index_new_name = 'edge_index.parquet'
edge_attribute_dist_new_name = 'edge_attribute_dist.parquet'
edge_attribute_charge_new_name = 'edge_attribute_charge.parquet'

# node_edge_path = os.path.join(data_3a_folder, "data_v1.0.0/nodes_edges")
# node_edge_path = f'{os_path}/nodes_edges_3A'

########################## Run all ###############################
df = pd.read_csv(args.directory_metadata)
for pdb_id in df.pdbID.values:
    pdb_path = os.path.join(args.directory_nodes_edges, pdb_id)
    node_feature = pd.read_parquet(os.path.join(pdb_path, node_feature_name))

    # Load CB and CA edge data
    cb_edge_index = pd.read_parquet(os.path.join(pdb_path, cb_edge_index_name))
    cb_edge_attribute_dist = pd.read_parquet(os.path.join(pdb_path, cb_edge_attribute_dist_name))
    cb_edge_attribute_charge = pd.read_parquet(os.path.join(pdb_path, cb_edge_attribute_charge_name))

    ca_edge_index = pd.read_parquet(os.path.join(pdb_path, ca_edge_index_name))
    ca_edge_attribute_dist = pd.read_parquet(os.path.join(pdb_path, ca_edge_attribute_dist_name))
    ca_edge_attribute_charge = pd.read_parquet(os.path.join(pdb_path, ca_edge_attribute_charge_name))

    # Find missing node indices in CB edges
    missing_index_src = [i for i in node_feature.index if i not in cb_edge_index.source.values]
    missing_index_tgt = [i for i in node_feature.index if i not in cb_edge_index.target.values]

    # Fill missing edges from CA edges
    fill_src_df = ca_edge_index[ca_edge_index.source.isin(missing_index_src)]
    fill_tgt_df = ca_edge_index[ca_edge_index.target.isin(missing_index_tgt)]
    fill_missing_index_src = fill_src_df.index
    fill_missing_index_tgt = fill_tgt_df.index
    fill_df = pd.concat([fill_src_df, fill_tgt_df]).drop_duplicates()
    fill_index = fill_missing_index_src.union(fill_missing_index_tgt)

    join_edge_index = pd.concat([cb_edge_index, fill_df])
    join_attribute_dist = pd.concat([cb_edge_attribute_dist, ca_edge_attribute_dist.iloc[fill_index]])
    join_attribute_charge = pd.concat([cb_edge_attribute_charge, ca_edge_attribute_charge.iloc[fill_index]])
    
    assert join_edge_index.shape[0] == join_attribute_dist.shape[0]
    assert join_edge_index.shape[0] == join_attribute_charge.shape[0]

    join_edge_index.to_parquet(os.path.join(pdb_path, edge_index_new_name))
    join_attribute_dist.to_parquet(os.path.join(pdb_path, edge_attribute_dist_new_name))
    join_attribute_charge.to_parquet(os.path.join(pdb_path, edge_attribute_charge_new_name))

    os.system(f'rm {os.path.join(pdb_path, '*_CA*')} {os.path.join(pdb_path, '*_CB*')} {os.path.join(pdb_path, '*-CA_*')} {os.path.join(pdb_path, '*-CB_*')}')

