print('''------ Epi4Ab ------
*** Prediction ***
''')

import torch
import os
from datetime import datetime

import_script_start = datetime.now()
from epitope_inference.arguments import initiate_argument
from epitope_inference.read_log import read_log
from epitope_inference.data_function import process_data
from source_code.run_setup.set_up import create_output_folder
from source_code.model.testing_function import test_model
from source_code.model.model_function import choose_model
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

print('Generate log: ', end='')
logging.log_result()
print('Done')