import pandas as pd
from source_code.data_function.data_function import batch_list

def process_data(logging):
    # NOTE need code for relaxed
    pdb_df = pd.read_csv(logging.directory_pdb_list)
    # if logging.use_relaxed:
    #     pdb_df = pdb_df + '_re'
    #     with open(logging.directory_relaxed_sequence, 'r') as f:
    #         sequence_data = json.load(f)
    # else:
    #     with open(logging.directory_sequence, 'r') as f:
    #         sequence_data = json.load(f)
    # if logging.use_relaxed:
    #     with open(logging.directory_relaxed_sequence, 'r') as f:
    #         relaxed_sequence_data = json.load(f)
    #     sequence_data = sequence_data | relaxed_sequence_data

    if logging.use_pretrained:      
        # agDict = {k:v["pdb_sequence"].replace('gap','').replace('x','') for k,v in sequence_data.items()}
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
    if logging.use_antiberty:
        from antiberty import AntiBERTyRunner
        ab_pretrained_model = AntiBERTyRunner()
        # relaxed_train_list = relaxed_train_df.to_numpy().flatten()
        # relaxed_train_data, adj_relaxed_train_list = batch_list(relaxed_train_list, logging, agDict, GTDict, logging.feature_name_dict, batchType='relaxed train',
        #                                                         pretrained_tokenize=pretrained_tokenize, pretrained_model=pretrained_model)
    pdb_list = pdb_df.to_numpy().flatten()
    test_data, new_pdb_list = batch_list(pdb_list, logging, batchType='test',
                        pretrained_tokenize=pretrained_tokenize, pretrained_model=pretrained_model,
                        ab_pretrained_model=ab_pretrained_model,
                        prediction=True)
    # test_data = batch_list(pdb_list, logging, agDict, featureNameDict=logging.feature_name_dict, batchType='test')
    return test_data, new_pdb_list