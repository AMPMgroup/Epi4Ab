import seaborn as sns
import matplotlib.pyplot as plt
import os

def result_loss(loss_df, logging):
    loss_df.to_parquet(os.path.join(logging.directory_output_folder, 'train_loss.parquet'))
    # train_loss = pd.read_csv(os.path.join(output_folder, 'train_loss.txt'))
    if logging.train_all in ['yes','with_validation']:
        sns.relplot(data = loss_df, x = 'epoch', y = 'loss', hue = 'type', kind = 'line')
    else:
        sns.relplot(data = loss_df, x = 'epoch', y = 'loss', hue = 'type', col = 'fold', col_wrap = 3, kind = 'line')
    if loss_df.loss.max() >= 1:
        plt.ylim(top = 1)
    plt.ylim(bottom = 0)
    plt.savefig(os.path.join(logging.directory_output_folder,'train_loss.png'))
    plt.close()

