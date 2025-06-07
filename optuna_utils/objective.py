from interface_prediction.model.training_function import TrainModel, choose_model, set_optimizer, subset_train_validate
from interface_prediction.model.testing_function import prediction_test, record_test
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import optuna
from optuna_utils.suggestion import trial_suggestion

class Objective:
    def __init__(self,train_data_raw, train_list_raw, logging, relaxed_train_data_raw=None, relaxed_train_list_raw=None, 
                     af_train_data_raw=None, adj_af_train_list=None):
        self.train_data_raw = train_data_raw
        self.train_list_raw = train_list_raw
        self.logging = logging 
        self.relaxed_train_data_raw=relaxed_train_data_raw
        self.relaxed_train_list_raw= relaxed_train_list_raw 
        self.af_train_data_raw=af_train_data_raw
        self.adj_af_train_list = adj_af_train_list

    def __call__(self, trial):
        trial_suggestion(trial, self.logging)
        train_class = TrainModel(self.logging.loss_function,
                                self.logging.cross_entropy_weight,
                                self.logging.batch_size,
                                self.logging.train_all,
                                self.logging.epoch_number,
                                self.logging.device,
                                self.logging.torch_seed)
        model = choose_model(self.logging).to(self.logging.device)
        optimizer = set_optimizer(model, self.logging)
        train_class.set_model(model, optimizer)
        # if logging.train_all == 'with_validation':
        train_ind, validate_ind = train_test_split(range(len(self.train_list_raw)), test_size = 0.1, random_state = self.logging.sklearn_seed)
        train_data, train_data_list, validate_data, validate_data_list = subset_train_validate(self.train_data_raw, self.train_list_raw, train_ind, validate_ind)
        if self.logging.use_relaxed:
            relaxed_train_data, relaxed_train_data_list, relaxed_validate_data, relaxed_validate_data_list = subset_train_validate(self.relaxed_train_data_raw, self.relaxed_train_list_raw, train_ind, validate_ind)
            train_data.extend(relaxed_train_data)
            train_data_list.extend(relaxed_train_data_list)
            validate_data.extend(relaxed_validate_data)
            validate_data_list.extend(relaxed_validate_data_list)
        if self.logging.use_alphafold:
            af_train_ind, af_validate_ind = train_test_split(range(len(self.adj_af_train_list)), test_size = 0.1, random_state = self.logging.sklearn_seed)
            af_train_data, af_train_data_list, af_validate_data, af_validate_data_list = subset_train_validate(self.af_train_data_raw, self.adj_af_train_list, af_train_ind, af_validate_ind)
            train_data.extend(af_train_data)
            train_data_list.extend(af_train_data_list)
            validate_data.extend(af_validate_data)
            validate_data_list.extend(af_validate_data_list)

        torch.manual_seed(self.logging.torch_seed)
        train_record = []

        # if self.train_all in ['yes','with_validation']:
        for epoch in range(self.logging.epoch_number):
            info_list = [epoch + 1]
            train_loss = train_class.train_model(train_data, info_list)
            train_record.extend(train_loss)
            record_data = 0
            model.eval()
            for pdb, data in zip(validate_data_list, validate_data):
                result = prediction_test(data, model, self.logging.device)
                pred_y = result.argmax(dim = 1)
                soft_y = F.softmax(result, dim=1)

                assert not data.y.isnan().any(), f'There is NaN value of pdb "{pdb}" in true Y {data.y}'
                assert not pred_y.isnan().any(), f'There is NaN value of pdb "{pdb}" in pred Y {pred_y}'
                assert not soft_y.isnan().any(), f'There is NaN value of pdb "{pdb}" in softmax Y {soft_y}'
                
                if self.logging.target_region == 'cips':
                    cips_record_list = record_test(data.y, pred_y, cips_evaluate = True )
                    record_data += cips_record_list[self.logging.target_metric]
                else:
                    record_list = record_test(data.y, pred_y, soft_y)
                    record_data += record_list[self.logging.target_metric]
            evaluation_result = record_data / len(validate_data_list)
            trial.report(evaluation_result, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return evaluation_result

