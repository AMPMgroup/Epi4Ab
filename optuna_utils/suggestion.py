def trial_suggestion(trial, logging):
    # logging.learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    # hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
    # num_layers = trial.suggest_int('num_layers', 1, 3)
    # optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    logging.train_all = 'with_validation'
    logging.num_layers = trial.suggest_int('num_layers', 5, 10) 
    logging.mha_num_layers = trial.suggest_int('mha_num_layers', 1, 10) 
    logging.batch_size = trial.suggest_int('batch_size', 32, 64,step=2)
    logging.drop_out= []
    logging.hidden_channel= []
    logging.learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, step=0.0005) 
    logging.weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, step=0.0005) 
    logging.use_deep_shallow = trial.suggest_categorical('use_deep_shallow', [True, False])
    logging.loss_function = trial.suggest_categorical('loss_function', ['cross_entropy', 'hce'])
    logging.shallow_layer = logging.num_layers - trial.suggest_int('shallow_layer_diff',1,logging.num_layers-1) 
    logging.shallow_cutoff = trial.suggest_float('shallow_cutoff', 1.5, 4.0 , step=0.1) 
    bond_att_weight = trial.suggest_uniform('bond_att_weight', -2, 2)
    lj_att_weight = trial.suggest_uniform('lj_att_weight', -2, 2)
    charge_att_weight = trial.suggest_uniform('charge_att_weight', -2, 2)
    logging.attribute_weight = [bond_att_weight, lj_att_weight, charge_att_weight]
    for i in range(logging.num_layers):
        hidden_channel = trial.suggest_int(f'hidden_channel_{i}', 32, 128, step=32) 
        logging.hidden_channel.append(hidden_channel)
        drop_out = trial.suggest_float(f'drop_out_{i}', 0, 0.5, step=0.1) 
        logging.drop_out.append(drop_out)
    # logging.optuna_target_region = 'cips'
    # logging.optuna_target_metric = 2
    
    # 0 recall_score
    # 1 precision_score,
    # 2 f1_score,
    # 3 accuracy_score,
    # 4 roc_auc_score,
    # 5 average_precision_score

