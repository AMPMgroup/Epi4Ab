import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_boxplot(test_plot_df, logging, preName, tagName):
    if logging.train_all in ['yes','with_validation']:
        sns.catplot(test_plot_df, x = 'value', col = 'type', hue = 'Label', col_wrap = 3, kind = 'box')
    else:
        sns.catplot(test_plot_df, x = 'value', y ='Fold', col = 'type', hue = 'Label', col_wrap = 3, kind = 'box')
    plt.xlim([-0.05,1.05])
    fileName = 'boxplot_' + preName + '_' + tagName + '.png'
    plt.savefig(os.path.join(logging.directory_output_folder, fileName))
    plt.close('all')

def result_boxplot(df, logging, resultRegion:bool=False):
    for pre_name in pd.unique(df.Type):
        type_df = df[df.Type == pre_name]
        if logging.train_all in ['yes','with_validation']:
            info_list = ['pdbId','Model','Label']
        else:
            info_list = ['pdbId','Fold','Model','Label']
            type_df = type_df.astype({'Fold':str})
        
        test_plot_df = type_df.melt(id_vars = info_list, 
                        value_vars = logging.evaluation_columns,
                        var_name='type', value_name='value')
        
        if resultRegion:
            tag_name = 'region'
        else:
            tag_name = 'whole'
        plot_boxplot(test_plot_df, logging, pre_name, tag_name)
        file_name = 'evaluation_' + pre_name + '_' + tag_name + '.parquet'
        test_plot_df.to_parquet(os.path.join(logging.directory_output_folder, file_name))