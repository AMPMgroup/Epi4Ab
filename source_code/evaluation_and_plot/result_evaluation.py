from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from tabulate import tabulate

def table_mean(df, logging, resultRegion:bool=True):
    scaler = StandardScaler()
    for pre_name in pd.unique(df.Type):
        type_df = df[df.Type == pre_name]
        if resultRegion:
            tag_name = 'region'
        else:
            tag_name = 'whole'
        file_name = 'evaluation_mean_' + pre_name + '_' + tag_name + '.txt'
        for label_name  in pd.unique(type_df.Label): #Label All or CIPS
            label_df = type_df[type_df.Label == label_name]

            evaluation_df = []
            if logging.train_all in ['yes','with_validation']:
                info_list = []
                scaler.fit(label_df[logging.evaluation_columns])
                result_mean = list(scaler.mean_)
                evaluation_df.append(result_mean)
            else:
                info_list = ['Fold']
                fold_result = []
                for fold_ind in pd.unique(label_df.Fold):
                    scaler.fit(label_df[label_df.Fold == fold_ind][logging.evaluation_columns])
                    result_mean = list(scaler.mean_)
                    evaluation_df.append([fold_ind] + result_mean)
                    fold_result.append(result_mean)
                scaler.fit(fold_result)
                fold_mean = list(scaler.mean_)
                evaluation_df.append(['Average'] + fold_mean)

            info_columns = info_list + logging.evaluation_columns
            with open(os.path.join(logging.directory_output_folder, file_name), 'a') as f:
                f.write(f'\n{label_name.upper()} result:\n')
                f.write(tabulate(evaluation_df, headers=info_columns))
                f.write('\n')