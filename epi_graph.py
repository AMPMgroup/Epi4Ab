print('''------ AMPM Project ------
*** Training Model ***
''')

import pandas as pd
import os
from datetime import datetime

print('Importing script')
import_script_start = datetime.now()
from interface_prediction.run_setup.logging import ModelLogging
from interface_prediction.run_setup.set_up import create_output_folder
from interface_prediction.run_setup.arguments import initiate_argument
from interface_prediction.data_function.data_function import process_data
from interface_prediction.model.training_function import process_training
from interface_prediction.evaluation_and_plot.result_evaluation import table_mean
from interface_prediction.evaluation_and_plot.loss import result_loss
import_script_time = datetime.now() - import_script_start
print(f'Import necessary library: {import_script_time}')

prepare_start_time = datetime.now()
args = initiate_argument()
# Prepare folder

if os.path.isdir(args.directory_output) == False:
    os.mkdir(args.directory_output)

model_logging = ModelLogging(args)
setup_logging_time = datetime.now() - prepare_start_time
print(f'Set up logging: {setup_logging_time}')

folder_name = str(model_logging.run_date) + '_' + model_logging.model_name
output_folder = create_output_folder(folder_name, model_logging.directory_output)
model_logging.directory_output_folder = output_folder
test_record_folder = os.path.join(output_folder,'test_record')
os.mkdir(test_record_folder)
model_logging.directory_test_record = test_record_folder
print(f'Results are located in {output_folder}.')
model_logging.prepare_time = datetime.now() - prepare_start_time

data_start_time = datetime.now()
if model_logging.train_all == 'yes':
    train_data, train_list = process_data(model_logging)
else:
    train_data, train_list, relaxed_train_data, relaxed_train_list, test_data, test_list = process_data(model_logging)
model_logging.data_time = datetime.now() - data_start_time

print('Start training')

train_start_time = datetime.now()
if model_logging.train_all == 'yes':
    train_loss_record, evaluation_record = process_training(train_data, train_list, model_logging)
else:
    train_loss_record, evaluation_record = process_training(train_data, train_list, model_logging, relaxed_train_data, relaxed_train_list, test_data, test_list)
model_logging.train_time = datetime.now() - train_start_time

result_start_time = datetime.now()
# Loss
if model_logging.train_all in ['yes','with_validation']:
    info_loss_columns = ['type', 'epoch', 'loss']
else:
    info_loss_columns = ['type', 'epoch', 'fold', 'loss']
loss_df = pd.DataFrame(train_loss_record, columns = info_loss_columns)
# Evaluation
if model_logging.train_all in ['yes','with_validation']:
    info_record_columns = ['pdbId', 'Model']
else:
    info_record_columns = ['pdbId','Fold','Model']
record_columns = info_record_columns + model_logging.evaluation_columns + ['Type'] + ['Label']
model_logging.info_record_columns = record_columns
evaluation_df = pd.DataFrame(evaluation_record, columns = record_columns)

print('Evaluating result.')
plotting_loss_start_time = datetime.now()
result_loss(loss_df, model_logging)
plotting_loss_time = datetime.now() - plotting_loss_start_time
print(f'Plotting loss: {plotting_loss_time}')

generate_table_start_time = datetime.now()
table_mean(evaluation_df, model_logging, resultRegion=model_logging.use_region)
generate_table_time = datetime.now() - generate_table_start_time
print(f'Summarize average result: {generate_table_time}')
model_logging.result_time = datetime.now() - result_start_time

model_logging.log_result()
print('Generate log: Done')

with open(os.path.join(output_folder,'parameters.txt'), 'w') as f:
    f.write(f'# Code version: {model_logging.code_version}\n')
    f.write(args.saved_params.replace(';','\n').replace('+',' '))
print('Saving parameters: Done')