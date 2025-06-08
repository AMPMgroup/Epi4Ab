print('''------ AMPM Project ------
*** Training Model ***
''')

import optuna
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as ov_m

print('Importing script')
import_script_start = datetime.now()
from interface_prediction.run_setup.logging import ModelLogging
from interface_prediction.run_setup.set_up import create_output_folder
from interface_prediction.run_setup.arguments import initiate_argument
from interface_prediction.data_function.data_function import process_data
from optuna_utils.objective import Objective
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
# print(f'Results are located in {output_folder}.')
model_logging.prepare_time = datetime.now() - prepare_start_time

data_start_time = datetime.now()
if model_logging.train_all == 'yes':
    train_data, train_list = process_data(model_logging)
else:
    train_data, train_list, relaxed_train_data, relaxed_train_list, af_train_data, adj_af_train_list, test_data, test_list = process_data(model_logging)
model_logging.data_time = datetime.now() - data_start_time

print('Start training')
if __name__ == "__main__":
    # if model_logging.train_all == 'yes':
    #     train_loss_record, evaluation_record = process_training(train_data, train_list, model_logging)
    # else:
    # train_loss_record, evaluation_record = process_optuna(train_data, train_list, model_logging, relaxed_train_data, relaxed_train_list, 
    #                                                         af_train_data, adj_af_train_list, test_data, test_list)
    objective = Objective(train_data, train_list, model_logging, relaxed_train_data, relaxed_train_list, 
                                                            af_train_data, adj_af_train_list)
    study = optuna.create_study(direction="maximize") # Maximize validation accuracy

    # Add a pruner to stop unpromising trials early
    # median_pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=3)
    # study = optuna.create_study(direction="maximize", pruner=median_pruner)

    print("Optimizing hyperparameters...")
    study.optimize(objective, n_trials=5) # Run 50 trials

    output_folder = os.path.join('.','optuna_utils','output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    study_output_folder = create_output_folder(str(model_logging.run_date), output_folder)

    complete_trials = 0
    pruned_trial = 0
    fail_trial = 0
    with open(os.path.join(study_output_folder,'optimization_result_all.txt'), 'w') as f:
        for trial in study.trials:
            if trial.state == 1:
                complete_trials += 1
                trial_state = 'Complete'
            elif trial.state == 2:
                pruned_trial += 1
                trial_state = 'Pruned'
            else:
                fail_trial += 1
                trial_state = 'Fail'
            f.write(f"Trial {trial.number}: State={trial_state}, Value={trial.value}, Params={trial.params}\n")
    trial = study.best_trial
    result_message = f'''\nOptimization finished.
Number of finished trials:  {len(study.trials)}
Number of complete trials:  {complete_trials}
Number of pruned trials:    {pruned_trial}
Number of fail trials:      {fail_trial}
Best trial:
    Value: {trial.value}
    Params:
'''
    # print("\nOptimization finished.")
    # print("Number of finished trials:", len(study.trials))
    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: ", trial.value)
    # print("  Params: ")
    for key, value in trial.params.items():
        result_message += ("    {}: {}\n".format(key, value))
    print(result_message)
    
    print(f'Results are located in {study_output_folder}.')
    
    with open(os.path.join(study_output_folder,'optimization_result.txt'),'w') as f:
        f.write(result_message)
    
    try:
        fig = ov_m.plot_optimization_history(study,target_name='F1_score')
        plt.tight_layout()
        plt.savefig(os.path.join(study_output_folder,"optimization_history.png"))
        plt.close(fig.figure) # Close the figure to free up memory
    except:
        print('optimization_history has error')
    try:
        fig = ov_m.plot_param_importances(study,target_name='F1_score')
        plt.tight_layout()
        plt.savefig(os.path.join(study_output_folder,"param_importances.png"))
        plt.close(fig.figure) # Close the figure to free up memory
    except:
        print('param_importances has error')
    try:
        fig = ov_m.plot_intermediate_values(study,target_name='F1_score')
        plt.tight_layout()
        plt.savefig(os.path.join(study_output_folder,"intermediate_values.png"))
        plt.close(fig.figure) # Close the figure to free up memory
    except:
        print('intermediate_values has error')
    try:
        fig = ov_m.plot_terminator_improvement(study,target_name='F1_score')
        plt.tight_layout()
        plt.savefig(os.path.join(study_output_folder,"terminator_improvement.png"))
        plt.close(fig.figure) # Close the figure to free up memory
    except:
        print('terminator_improvement has error')
    try:
        fig = ov_m.plot_parallel_coordinate(study,target_name='F1_score')
        plt.tight_layout()
        plt.savefig(os.path.join(study_output_folder,"parallel_coordinate.png"))
        plt.close(fig.figure) # Close the figure to free up memory
    except:
        print('parallel_coordinate has error')