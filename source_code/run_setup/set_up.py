import os
import warnings

def create_output_folder(folder_name, parent_folder):
    file_count = 1
    for folder in os.listdir(parent_folder):
        if folder_name in folder:
            file_count += 1
    output_folder = os.path.join(parent_folder,folder_name + '_' + str(file_count))
    os.mkdir(output_folder)
    return output_folder

def multilayer_warning(num_layers, target_var, var_name:str):
    if num_layers <= 0:
        raise ValueError(f'{num_layers} can not be less than or equal 0.')
    elif (len(target_var) > 1) & (num_layers > len(target_var)):
        raise ValueError(f'There are more layers than number of {var_name}. Number of layers {num_layers} vs input {var_name} {target_var}')
    elif num_layers < len(target_var):
        unused_list = [target_var[i] for i in range(len(target_var)) if i >= num_layers]
        warnings.warn(f'There are less layers than number of {var_name}. {unused_list} are not used.')
