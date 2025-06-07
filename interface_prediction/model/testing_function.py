import numpy as np
import pandas as pd
import os
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
from tqdm import tqdm
from interface_prediction.evaluation_and_plot.networkx import plot_network

def prediction_test(data, modelBuild, device):
    data.to(device)
    result = modelBuild(data.x, data.x_seq, data.edge_index, data.edge_attr, data.x_ab, 
                        data.ab_padding_mask, data.feature_token, data.node_size)
    # trueY = data.y == 0
    return result

def record_test(trueY, predY, softY=None, cips_evaluate = False):
    true_interface, pred_interface = trueY.detach().cpu().numpy(), predY.detach().cpu().numpy()
    if cips_evaluate:
        true_interface = np.where(true_interface == 1, True, False)
        pred_interface = np.where(pred_interface == 1, True, False)
        roc_auc_score = sk_metrics.roc_auc_score(true_interface,pred_interface)
        average_precision_score = sk_metrics.average_precision_score(true_interface,pred_interface)
        recall_score = sk_metrics.recall_score(true_interface,pred_interface, zero_division=0, average='binary')
        precision_score = sk_metrics.precision_score(true_interface,pred_interface, zero_division=0, average='binary')
        f1_score = sk_metrics.f1_score(true_interface,pred_interface, zero_division=0, average='binary')
        accuracy_score = sk_metrics.accuracy_score(true_interface,pred_interface)
    else:
        if len(np.unique(true_interface)) == 2:
            true_interface = true_interface != 0
            pred_interface = pred_interface != 0
            roc_auc_score = sk_metrics.roc_auc_score(true_interface,pred_interface)
            average_precision_score = sk_metrics.average_precision_score(true_interface,pred_interface)
        else:
            if softY is not None:
                soft_pred_interface = softY.detach().cpu().numpy()
                roc_auc_score = sk_metrics.roc_auc_score(true_interface,soft_pred_interface, average='macro', multi_class='ovo')
                average_precision_score = sk_metrics.average_precision_score(true_interface,soft_pred_interface, average='micro')
            else:
                roc_auc_score = sk_metrics.roc_auc_score(true_interface,pred_interface, average='macro', multi_class='ovo')
                average_precision_score = sk_metrics.average_precision_score(true_interface,pred_interface, average='micro')
            '''
            For example recall scrore
                Micro: Sum of absolute tp and fn/fp of each classes.
                Macro: Average of recall of all classes.
            '''
        # Use Micro cause average might make the result biased towards outliers in case of imbalanced data. 
        recall_score = sk_metrics.recall_score(true_interface,pred_interface, zero_division=0, average='macro')
        precision_score = sk_metrics.precision_score(true_interface,pred_interface, zero_division=0, average='macro')
        f1_score = sk_metrics.f1_score(true_interface,pred_interface, zero_division=0, average='macro')
        accuracy_score = sk_metrics.accuracy_score(true_interface,pred_interface)

    return [recall_score,
            precision_score,
            f1_score,
            accuracy_score,
            roc_auc_score,
            average_precision_score]
    
def record_res_id(predY, trueY, softY, resID, resShort, pdb, fold_folder):
    df = pd.DataFrame({'res_id':resID,
                       'true_y':trueY,
                       'pred_y':predY,
                       'prob_gt':softY[np.arange(len(softY)),trueY],
                       'prob_0':softY[:,0],
                       'prob_1':softY[:,1],
                       'prob_2':softY[:,2]})
    # df.to_parquet(os.path.join(fold_folder, f'{pdb}.parquet'))
    df.to_csv(os.path.join(fold_folder, f'{pdb}.txt'), sep='\t', index=False)  # Save as txt file
    score = np.log(softY/(1-softY + 1e-6))
    final_df = pd.DataFrame({'res_id':resID,
                            'res_name':resShort,
                            'pred_label':predY,
                            'prob.':softY[np.arange(len(softY)),predY],
                            'score':score[np.arange(len(score)),predY]})
    final_df.to_csv(os.path.join(fold_folder, f'{pdb}_final_result.txt'), sep='\t', index=False)  # Save as txt file

def record_test_pdb(data, predY, softY, res_short, pdb, fold_folder, plot_network_check, networkx_seed):
    true_y = data.y.detach().cpu().numpy()
    pred_y = predY.detach().cpu().numpy()
    soft_y = softY.detach().cpu().numpy()
    res_id = data.res_id.detach().cpu().numpy()
    record_res_id(pred_y, true_y, soft_y, res_id, res_short, pdb, fold_folder)
    if plot_network_check:
        plot_network(res_id, pred_y, true_y, data.edge_index, data.edge_attr, pdb, fold_folder, networkx_seed)

@torch.no_grad()
def test_model(modelBuild, testData, testList, logging, testType:str='train', foldInd=None):
    '''
    testType = ['train', 'validate','test']
    '''
    if logging.train_all in ['yes','with_validation']:
        info_list = [logging.model_name]
        fold_folder = logging.directory_test_record
        iter_desc = f'Testing {testType} pdbs'
    else:
        info_list = [foldInd, logging.model_name]
        fold_folder = os.path.join(logging.directory_test_record, f'fold_{foldInd}')
        if not os.path.exists(fold_folder):
            os.mkdir(fold_folder)
        iter_desc = f'Testing {testType} pdbs, fold {foldInd}'
    record_data = []
    modelBuild.eval()
    tqdm_enum = tqdm(zip(testList, testData), total=len(testList), desc = iter_desc, unit='pdb')
    if logging.loss_function in ['cross_entropy','hce']:
        for pdb, data in tqdm_enum:
            result = prediction_test(data, modelBuild, logging.device)
            pred_y = result.argmax(dim = 1)
            soft_y = F.softmax(result, dim=1)

            assert not data.y.isnan().any(), f'There is NaN value of pdb "{pdb}" in true Y {data.y}'
            assert not pred_y.isnan().any(), f'There is NaN value of pdb "{pdb}" in pred Y {pred_y}'
            assert not soft_y.isnan().any(), f'There is NaN value of pdb "{pdb}" in softmax Y {soft_y}'
            record_list = record_test(data.y, pred_y, soft_y)
            record_list = [pdb] + info_list + record_list + [testType] + ['all']
            record_data.append(record_list)
            # CIPS only evaluation
            cips_record_list = record_test(data.y, pred_y, cips_evaluate = True )
            cips_record_list = [pdb] + info_list + cips_record_list + [testType] + ['cips']
            record_data.append(cips_record_list)
            if testType == 'test':
                record_test_pdb(data, pred_y, soft_y, data.res_short, pdb, fold_folder, logging.plot_network, logging.networkx_seed)
      
    elif logging.loss_function == 'mse':
        for pdb, data in tqdm_enum:
            result = prediction_test(data, modelBuild, logging.device)
            true_y = data.y == 0
            pred_y = result.reshape(-1) <= logging.mse_threshold
            assert not data.y.isnan().any(), f'There is NaN value of pdb "{pdb}" in true Y {data.y}'
            assert not pred_y.isnan().any(), f'There is NaN value of pdb "{pdb}" in pred Y {pred_y}'
            record_list = record_test(true_y, pred_y)
            record_list = [pdb] + info_list + record_list + [testType]
            record_data.append(record_list)
            if testType == 'test':
                record_test_pdb(data, pred_y, pdb, fold_folder, logging)
                
    return record_data