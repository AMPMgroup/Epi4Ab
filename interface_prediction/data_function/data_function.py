import pandas as pd
import os
import torch
import json
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
import pandas.api.types as ptypes
from .attribute import CalculateAttribute
from .aa_tokenizer import AATokenizer
from .extraction import extract_feature, filter_neighbor

def batch_list(pdbList, logging, featureNameDict={}, batchType=None, pretrained_tokenize=None, pretrained_model=None, 
               ab_pretrained_model=None):
    '''
    This function gather information of structure and sequence data:
    - feature data: structure and/or sequence data
    - label data
    - attribute data
    - edge: neighbor nodes
    Input: list of PDBs.
    Output: torch-geometric data.
    '''
    dataBatch = []
    error_list = []
    new_pdb_list = []

    # logging.feature_file = 'pdb_profile.parquet'
    logging.feature_file = 'node_feature.parquet'
    logging.edge_attribute_file = 'edge_attribute_dist.parquet'
    logging.edge_attribute_charge_file = 'edge_attribute_charge.parquet'
    logging.edge_index_file = 'edge_index.parquet'
    logging.node_label_file = 'node_label_pi.parquet'

    # Attribute
    calc_attribute = CalculateAttribute(logging.device,
                                        logging.edge_type,
                                        logging.gradient_attribute,
                                        logging.attribute_no_bond,
                                        logging.attribute_no_lj,
                                        logging.attribute_no_charge,
                                        logging.attribute_weight,
                                        logging.bias_distance)

    if logging.softmax_data:
        softmax_class = torch.nn.Softmax(dim=1)

    if logging.use_token:
        aa_tokenizer = AATokenizer(featureNameDict, logging.ab_onehot_vh_columns, logging.ab_onehot_vl_columns)

    for pdbId in tqdm(pdbList, desc=f'Processing {batchType} data', unit='pdbId'):
        # try:
        if logging.use_relaxed & (pdbId[-3:] == '_re'): # Get data based on relaxed or bound
        #     pdb_folder = os.path.join(logging.directory_data, pdbId[:-3])
            label_folder = os.path.join(logging.directory_data, pdbId[:-3])
        else:
        #     pdb_folder = os.path.join(logging.directory_data, pdbId)
            label_folder = os.path.join(logging.directory_data, pdbId)
        feature_folder = os.path.join(logging.directory_processed_data, pdbId)
        # featureData = pd.read_parquet(os.path.join(feature_folder, logging.feature_file))
        pdb_folder = os.path.join(logging.directory_data, pdbId)
        featureData = pd.read_parquet(os.path.join(pdb_folder, logging.feature_file))
        # featureData = featureData[featureData.chainType == 'antigen']
        assert not featureData.empty, f'Feature of pdb {pdbId} is empty.'
        edgeIndex = pd.read_parquet(os.path.join(pdb_folder, logging.edge_index_file))
        assert not edgeIndex.empty, f'Edge index of pdb {pdbId} is empty.'
        edgeAttribute = pd.read_parquet(os.path.join(pdb_folder, logging.edge_attribute_file)).astype(float)
        assert not edgeAttribute.empty, f'Edge attribute distance of pdb {pdbId} is empty.'
        edgeCharge = pd.read_parquet(os.path.join(pdb_folder, logging.edge_attribute_charge_file))
        assert not edgeCharge.empty, f'Edge attribute charge of pdb {pdbId} is empty.'
        nodeLabel = pd.read_parquet(os.path.join(label_folder, logging.node_label_file)).astype(int)
        assert ptypes.is_float_dtype(edgeAttribute['dist']), f'Attribute of pdb {pdbId} is not float.'
        edgeAttribute = edgeAttribute.abs()

        # Label
        label = torch.tensor(nodeLabel.isInterface.astype(int).to_numpy().T, dtype=torch.long)
        # Edge
        edge = torch.tensor(edgeIndex.to_numpy().T)
        # Attribute
        attribute = torch.tensor(edgeAttribute.to_numpy().reshape(-1), dtype = torch.float)
        attributeCharge = torch.tensor(edgeCharge.to_numpy().reshape(-1), dtype = torch.int)
        # Filter neighbor
        if logging.edge_threshold < 10:
            edge, attribute, attributeCharge = filter_neighbor(edge, attribute, attributeCharge, logging.edge_threshold)
        attribute = calc_attribute(attribute, attributeCharge)
        assert not attribute.isnan().any(), f'There is nan in attribute {attribute}'
        res_id = torch.tensor(featureData.resId.to_numpy(), dtype=torch.long)

        if logging.use_pretrained:
            with open(os.path.join(logging.directory_processed_data, pdbId, 'sequence','antigen_sequence.json'), 'r') as f:
                x_seq = json.load(f)['pdb_sequence'].replace('gap','').replace('x','')
            if logging.freeze_pretrained:
                encode_input = pretrained_tokenize(x_seq, return_tensors = 'pt')
                output = pretrained_model(**encode_input)
                x_seq = output[0].reshape(output[0].shape[1],-1)[1:-1,:].detach()
        else:
            x_seq = torch.tensor(0)

        if logging.ab_feature_input:
            featureData[logging.ab_continuous_columns_change] = logging.ab_continuous_columns_value
            featureData[logging.ab_onehot_vh_columns] = logging.ab_onehot_vh_columns_value
            featureData[logging.ab_onehot_vl_columns] = logging.ab_onehot_vl_columns_value

        if logging.use_token:
            # res_short_list = featureData.resShort.to_list()
            vh_df = featureData[logging.ab_onehot_vh_columns]
            vh_fam = vh_df.columns[(vh_df == 1).all()].item()
            vl_df = featureData[logging.ab_onehot_vl_columns]
            vl_fam = vl_df.columns[(vl_df == 1).all()].item()
            feature_token = aa_tokenizer.tokenize_feature(vh_fam, vl_fam)
        else:
            feature_token = torch.tensor(0)

        if logging.use_struct:
            feature_struct = extract_feature(featureData, pdbId, logging)
            assert not torch.isnan(feature_struct).any(), f'Feature struct of {pdbId} has nan'
            assert feature_struct.size(dim=0) == label.size(dim=0), f'PDB {pdbId} has mismatch size of feature {feature_struct.size()} vs. label {label.size()}'
            if logging.softmax_data:
                feature_struct = softmax_class(feature_struct)
        else:
            feature_struct = torch.tensor(0)

        if logging.use_antiberty:
            with open(os.path.join(feature_folder, 'sequence','cdr_sequence.json'), 'r') as f:
                ab_feature = ab_pretrained_model.embed([json.load(f)['H3_seq']])[0].detach().cpu()
            ab_feature = torch.cat((ab_feature, torch.zeros(logging.antiberty_max_len - ab_feature.size(0), 512)),0).flatten()
            ab_feature = ab_feature.expand(feature_struct.size(0), -1)
            feature_struct = torch.cat((feature_struct, ab_feature), dim=1)
        
        assert edge.size(dim=1) == attribute.size(dim=0), f'PDB {pdbId} has mismatch size of edge {edge.size()} vs. attribute {label.size()}'

        res_short = featureData.resShort.to_numpy()
        new_pdb_list.append(pdbId)
        dataBatch.append(Data(x=feature_struct,
                                x_seq=x_seq,
                            y=label, 
                            edge_index=edge,
                            edge_attr=attribute,
                            feature_token=feature_token,
                            res_id=res_id,
                            res_short=res_short,
                            pdb_id=pdbId,
                            node_size=torch.tensor([res_id.size(dim=0)])))
        del feature_struct
        del label
        del edge
        del attribute
        del attributeCharge
        del res_id
        # except:
        #     error_list.append(pdbId)
        
    if error_list:
        print(f'Error PDB while processing data: {error_list}')
        logging.error_data_process_list.extend(error_list)
    return dataBatch, new_pdb_list

def read_feature_name(logging):
    with open(Path(__file__).parent / 'feature_columns.json', 'r') as f:
        feature_name_dict = json.load(f)
    feature_name_dict = feature_name_dict[logging.feature_version]
    logging.continuous_columns = feature_name_dict['Ag_features']['continuous']
    logging.ag_continuous_columns = feature_name_dict['Ag_features']['continuous']
    logging.onehot_columns = feature_name_dict['Ag_features']['onehot']
    logging.ag_onehot_columns = feature_name_dict['Ag_features']['onehot']
    if len(feature_name_dict) >= 2:
        logging.ab_continuous_columns = feature_name_dict['Ab_features']['continuous']
        logging.ab_onehot_vh_columns = feature_name_dict['Ab_features']['onehot_vh']
        logging.ab_onehot_vl_columns = feature_name_dict['Ab_features']['onehot_vl']
        logging.amino_acid_list = feature_name_dict['AA_short']
        logging.continuous_columns = logging.continuous_columns + logging.ab_continuous_columns
        logging.reserved_columns = len(logging.continuous_columns) + len(logging.onehot_columns)
        if logging.feature_version == 'v1.0.1':
            logging.onehot_columns = logging.onehot_columns + logging.ab_onehot_vh_columns + logging.ab_onehot_vl_columns
            if logging.use_struct:
                logging.feature_columns_number = logging.reserved_columns + len(logging.ab_onehot_vh_columns) + len(logging.ab_onehot_vl_columns)
            
        elif logging.feature_version == 'v1.0.2':

            if logging.use_token:
                token_dim = logging.token_dim * 2 # For vh and vl
            else:
                logging.onehot_columns = logging.onehot_columns + logging.ab_onehot_vh_columns + logging.ab_onehot_vl_columns
                token_dim = len(logging.ab_onehot_vh_columns) + len(logging.ab_onehot_vl_columns)

            logging.feature_columns_number = logging.reserved_columns + token_dim
    return feature_name_dict

def process_data(logging):
    '''
    This function split the dataset to train-validate and test.
    Output: train_validate batch, test batch and list of test pdb for model evaluation.
    '''

    if logging.use_pretrained:
        # agDict = {k:v["pdb_sequence"].replace('gap','').replace('x','') for k,v in sequence_data.items()}
        # Pre-trained
        if logging.freeze_pretrained:
            if logging.pretrained_model == 'protBERT':
                from transformers import BertModel, BertTokenizer
                pretrained_tokenize = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
                pretrained_model = BertModel.from_pretrained("Rostlab/prot_bert")
            elif logging.pretrained_model[:4] == 'ESM2':
                from transformers import AutoTokenizer, EsmModel
                if logging.pretrained_model == 'ESM2_t6':
                    pretrained_tokenize = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
                    pretrained_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
                elif logging.pretrained_model == 'ESM2_t12':
                    pretrained_tokenize = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
                    pretrained_model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
                elif logging.pretrained_model == 'ESM2_t30':
                    pretrained_tokenize = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
                    pretrained_model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
                elif logging.pretrained_model == 'ESM2_t33':
                    pretrained_tokenize = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
                    pretrained_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
                elif logging.pretrained_model == 'ESM2_t36':
                    pretrained_tokenize = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
                    pretrained_model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
                pretrained_model.eval()
        else:
            pretrained_tokenize = None
            pretrained_model = None

        # with open(Path(__file__).parent / 'pretrained_parameter.json', 'r') as f:
        #     pretrained_para = json.load(f)
    logging.feature_name_dict = read_feature_name(logging)

    logging.in_feature += logging.pretrained_dim if logging.use_pretrained else 0
    logging.in_feature += logging.feature_columns_number
    assert logging.in_feature != 0, 'Dimension of feature could not be 0.'

    if logging.use_antiberty:
        from antiberty import AntiBERTyRunner
        ab_pretrained_model = AntiBERTyRunner()
        logging.in_feature += logging.antiberty_max_len * 512
    else:
        ab_pretrained_model = None
    
    pdb_df = pd.read_csv(logging.directory_pdb_list)
    train_list = pdb_df.to_numpy().flatten()
    
    if logging.train_all == 'yes':
        if logging.use_relaxed:
            pdb_df = pd.concat([pdb_df, pdb_df + '_re'])
        train_list = pdb_df.to_numpy().flatten()
        train_data, adj_train_list = batch_list(train_list, logging, logging.feature_name_dict, batchType='train', 
                                                pretrained_tokenize=pretrained_tokenize, pretrained_model=pretrained_model,
                                                ab_pretrained_model=ab_pretrained_model)
        return train_data, adj_train_list
    else:
        train_data, adj_train_list = batch_list(train_list, logging, logging.feature_name_dict, batchType='train', 
                                                pretrained_tokenize=pretrained_tokenize, pretrained_model=pretrained_model,
                                                ab_pretrained_model=ab_pretrained_model)
        if logging.use_relaxed:
            relaxed_train_df = pdb_df + '_re'
            relaxed_train_list = relaxed_train_df.to_numpy().flatten()
            relaxed_train_data, adj_relaxed_train_list = batch_list(relaxed_train_list, logging, logging.feature_name_dict, batchType='relaxed train',
                                                                    pretrained_tokenize=pretrained_tokenize, pretrained_model=pretrained_model,
                                                                    ab_pretrained_model=ab_pretrained_model)
        else:
            adj_relaxed_train_list = None
            relaxed_train_data = None
        test_list = pd.read_csv(logging.directory_unseen_pdb_list).to_numpy().flatten()
        test_data, test_list = batch_list(test_list, logging, logging.feature_name_dict, batchType='test',
                                          pretrained_tokenize=pretrained_tokenize, pretrained_model=pretrained_model,
                                          ab_pretrained_model=ab_pretrained_model)
        return train_data, adj_train_list, relaxed_train_data, adj_relaxed_train_list, test_data, test_list