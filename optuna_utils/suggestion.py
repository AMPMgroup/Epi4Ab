def trial_suggestion(trial, logging):
    # logging.learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    # hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
    # num_layers = trial.suggest_int('num_layers', 1, 3)
    # optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    logging.train_all = 'with_validation'
    logging.num_layers = trial.suggest_int('num_layers', 1, 3) 
    logging.drop_out= []
    logging.hidden_channel= []
    for i in range(logging.num_layers):
        hidden_channel = trial.suggest_int(f'hidden_channel_{i}', 8, 32, step=8) 
        logging.hidden_channel.append(hidden_channel)
        drop_out = trial.suggest_float(f'drop_out_{i}', 0, 0.5, step=0.1) 
        logging.drop_out.append(drop_out)
    logging.target_region = 'cips'
    logging.target_metric = 2
    
    # 0 recall_score
    # 1 precision_score,
    # 2 f1_score,
    # 3 accuracy_score,
    # 4 roc_auc_score,
    # 5 average_precision_score