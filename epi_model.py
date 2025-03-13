print('''------ AMPM Project ------
*** Testing Model ***
''')

import torch
import os
import pandas as pd
from datetime import datetime
from tabulate import tabulate

import_script_start = datetime.now()
from use_model.arguments import initiate_argument
from use_model.read_log import read_log
from use_model.data_function import process_data
from interface_prediction.run_setup.set_up import create_output_folder
from interface_prediction.model.testing_function import test_model
from interface_prediction.evaluation_and_plot.boxplot import result_boxplot
from interface_prediction.evaluation_and_plot.result_evaluation import table_mean
from interface_prediction.model.model_function import choose_model
import_script_time = datetime.now() - import_script_start
print(f'Import necessary library: {import_script_time}')

# Initiate arguments
prepare_start_time = datetime.now()
args = initiate_argument()

# Create output folder
if os.path.isdir(args.directory_output) == False:
    os.mkdir(args.directory_output)

# Read model's log.md
logging = read_log(args)
setup_logging_time = datetime.now() - prepare_start_time
print(f'Set up logging: {setup_logging_time}')

# Load model
load_model_start_time = datetime.now()
if logging.save_not_as_statedict:
    model = torch.load(os.path.join(logging.directory_model_folder, 'model.pt'))
else:
    model = choose_model(logging).to(logging.device)
    model.load_state_dict(torch.load(os.path.join(logging.directory_model_folder, 'model.pt'), weights_only=True))
model.to(logging.device)
load_model_time = datetime.now() - load_model_start_time
print(f'Loading model: {load_model_time}')

# Create run session output folder
folder_name = str(logging.run_date) + '_' + logging.model_name
output_folder = create_output_folder(folder_name, logging.directory_output)
logging.directory_output_folder = output_folder
test_record_folder = os.path.join(output_folder,'test_record')
os.mkdir(test_record_folder)
logging.directory_test_record = test_record_folder
print(f'Results are located in {output_folder}.')
logging.prepare_time = datetime.now() - prepare_start_time

# Process test data
data_start_time = datetime.now()
test_data, test_list = process_data(logging)
logging.data_time = datetime.now() - data_start_time

# Use model
test_start_time = datetime.now()
evaluation_record = test_model(model, test_data, test_list, logging, testType='test')
logging.test_time = datetime.now() - test_start_time

# Convert evaluation data to pandas
result_start_time = datetime.now()
evaluation_df = pd.DataFrame(evaluation_record, columns = logging.info_record_columns)

plotting_boxplot_start_time = datetime.now()
result_boxplot(evaluation_df, logging, resultRegion = logging.use_region)
plotting_boxplot_time = datetime.now() - plotting_boxplot_start_time
print(f'Plotting boxplot: {plotting_boxplot_time}')

generate_table_start_time = datetime.now()
table_mean(evaluation_df, logging, resultRegion = logging.use_region)
generate_table_time = datetime.now() - generate_table_start_time
print(f'Summarize average result: {generate_table_time}')


with open(os.path.join(output_folder, 'evaluation_each_pdb.txt'), 'w') as f:
        f.write(tabulate(evaluation_record, headers=logging.info_record_columns))
        f.write('\n')
logging.result_time = datetime.now()

print('Generate log: ', end='')
logging.log_result()
print('Done')