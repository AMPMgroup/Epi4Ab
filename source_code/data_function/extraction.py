import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def extract_feature(featureData,
                    pdb_id, 
                    logging):
    '''
    This function extract the feature from feature file in node-edge.
    It normalized the continuous data, not one-hot.
    Input: feature data (structure data)
    Output: tensor data structure
    '''
    check_col = [col_name for col_name in logging.continuous_columns + logging.onehot_columns if col_name not in featureData.columns]
    assert check_col == [], f'{check_col} is not in feature columns of {pdb_id}'
    if 'charge' in featureData.columns:
        featureData = featureData.astype({'charge' : int})

    if logging.softmax_data | logging.not_normalize_data:
        feature_data = featureData[logging.continuous_columns]
    else:
        # Separate ag continuous and ab continuous
        # Ag continuous is normalized
        # Ab len is divided by 100
        # Ab score is reserved
        ag_continuous_data = np.array(StandardScaler().fit_transform(featureData[logging.ag_continuous_columns]))
        len_columns = [col for col in logging.ab_continuous_columns if col not in ['H3_score','L1_score']]
        score_columns = [col for col in logging.ab_continuous_columns if col in ['H3_score','L1_score']]
        ab_continuous_data = (featureData[len_columns]/100).to_numpy()
        feature_data = np.concatenate((ag_continuous_data,ab_continuous_data,featureData[score_columns].to_numpy()), axis=1)
    feature_data = np.concatenate((feature_data , featureData[logging.onehot_columns].to_numpy()), axis = 1)
    # assert feature_data.shape[1] == logging.feature_columns_number, f'''Shape is not match: 
    # result shape: {feature_data.shape[1]} vs. column length: {logging.feature_columns_number}'''
    return torch.tensor(feature_data, dtype = torch.float)

def extract_seq_feature(sequence, 
                        resID, 
                        pdb_folder, 
                        tokenizer,
                        bert_model,
                        use_region):
    '''
    This function utilize the pre-trained tokenizer and model to generate feature for sequence.
    Input: protein sequence
    Output: tensor data sequence
    '''
    encode_input = tokenizer(sequence, return_tensors = 'pt')
    # print(sequence)
    # print(encode_input)
    # assert sequence is None, 'sfd'
    output = bert_model(**encode_input)
    feature_data = output[0].reshape(output[0].shape[1],-1)[1:-1,:].cpu().detach().numpy()

    if use_region:
        resId_list = resID.tolist()
        node_list = pd.read_parquet(os.path.join(pdb_folder, 'node_list.parquet'))
        indices_list = [node_list.index[node_list['resId'] == val].tolist()[0] for val in resId_list]
        feature_data = feature_data[indices_list]

    return torch.tensor(feature_data, dtype = torch.float)

def filter_neighbor(edge, dist_att, charge_att, dist_threshold):
    filter_bool = dist_att <= dist_threshold
    edge = edge[:,filter_bool]
    dist_att = dist_att[filter_bool]
    charge_att = charge_att[filter_bool]
    return edge, dist_att, charge_att