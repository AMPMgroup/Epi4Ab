import os
import json
from .logging import UseModelLogging, ModelParamsLogging

def read_log(args, model_params = False):
    log_path = args.directory_model_folder
    if os.path.isdir(log_path) == False:
        raise ValueError(f'{log_path} does not exist.')
    with open(os.path.join(log_path, 'log.json'), 'r') as f:
        log_dict = json.load(f)
    if model_params:
        return ModelParamsLogging(log_dict, args)
    else:
        return UseModelLogging(log_dict, args)