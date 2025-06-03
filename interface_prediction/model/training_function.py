import torch
import os
from tqdm import trange
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from .testing_function import test_model
from .model_function import choose_model
from .optimizer import set_optimizer
from .loss_function import get_loss_function

class TrainModel:
    def __init__(self,
                 loss_function,
                 cross_entropy_weight,
                 batch_size,
                 train_all,
                 epoch_number,
                 device,
                 torch_seed):
        self.modelBuild = None
        self.optimizer = None
        self.batch_size = batch_size
        self.train_all = train_all
        self.epoch_number = epoch_number
        self.device = device
        self.torch_seed = torch_seed
        self.loss_function_name = loss_function
        self.loss_function = get_loss_function(loss_function, 
                                               cross_entropy_weight).to(self.device)

    def set_model(self, modelBuild, optimizer):
        self.modelBuild = modelBuild
        self.optimizer = optimizer

    def process_modelling(self, data):
        data = data.to(self.device)
        out = self.modelBuild(data.x, data.x_seq, data.edge_index, data.edge_attr, data.x_ab, data.feature_token, data.node_size)
        assert out.shape[0] == data.y.shape[0], f'Model output and Ground truth have mismatch shape {out.shape} vs. {data.y.shape}, PDBs in batch: {data.pdb_id}'
        loss = self.loss_function(out, data.y)
        assert not loss.isnan().any(), f'There is NaN value after loss function {loss}'
        item_loss = [loss.item()]
        del out
        return loss, item_loss
    
    def train_model(self, trainData, infoList):
        '''
        This is the function used for training model.
        Input: model, loss function, optimizer, device.
        Output: loss from training.
        '''
        self.modelBuild.train()
        lossList = []
        trainLoader = DataLoader(trainData, batch_size=self.batch_size, shuffle = True)
        
        for data in trainLoader:
            self.optimizer.zero_grad()
            loss, item_loss = self.process_modelling(data)
            
            loss.backward()
            self.optimizer.step()
            lossList.append(['train'] + infoList + item_loss)

            # Free GPU memory
            del loss
            torch.cuda.empty_cache()
        return lossList

    @torch.no_grad()
    def validate_model(self, testData, infoList):
        '''
        This is the function used for validating model.
        Input: model, loss function, optimizer, device.
        Output: loss from validating.
        '''
        lossList = []
        self.modelBuild.eval()

        for data in testData:
            _, item_loss = self.process_modelling(data)
            lossList.append(['validate'] + infoList + item_loss)
            torch.cuda.empty_cache()
        return lossList

    def epoch_training(self, train_data, fold_ind=None, validate_data=None):
        torch.manual_seed(self.torch_seed)
        train_record = []

        if self.train_all in ['yes','with_validation']:
            for epoch in trange(self.epoch_number, desc = f'Training', unit='epoch'):
                info_list = [epoch + 1]
                train_loss = self.train_model(train_data, info_list)
                train_record.extend(train_loss)
                if self.train_all == 'with_validation':
                    validate_loss = self.validate_model(validate_data, info_list)
                    train_record.extend(validate_loss)
                    del validate_loss
                del train_loss
                
        else:
            for epoch in trange(self.epoch_number, desc = f'Training fold {fold_ind + 1}', unit='epoch'):
                info_list = [epoch + 1, fold_ind + 1]
                train_loss = self.train_model(train_data, info_list)
                train_record.extend(train_loss)
                validate_loss = self.validate_model(validate_data, info_list)
                train_record.extend(validate_loss)
                del train_loss
                del validate_loss
                
        return  train_record

def subset_train_validate(train_data_raw, train_list_raw, train_ind, validate_ind):
    train_data = [train_data_raw[ind] for ind in train_ind]
    train_data_list = [train_list_raw[ind] for ind in train_ind]
    validate_data = [train_data_raw[ind] for ind in validate_ind]
    validate_data_list = [train_list_raw[ind] for ind in validate_ind]
    return train_data, train_data_list, validate_data, validate_data_list

def process_training(train_data_raw, train_list_raw, logging, relaxed_train_data_raw=None, relaxed_train_list_raw=None, 
                     af_train_data_raw=None, adj_af_train_list=None, test_data=None, test_list=None):
    # torch.autograd.set_detect_anomaly(True)
    loss_record = []
    evaluation_record = []
    train_class = TrainModel(logging.loss_function,
                            logging.cross_entropy_weight,
                            logging.batch_size,
                            logging.train_all,
                            logging.epoch_number,
                            logging.device,
                            logging.torch_seed)
    if logging.train_all in ['yes','with_validation']:
        model = choose_model(logging).to(logging.device)
        optimizer = set_optimizer(model, logging)
        train_class.set_model(model, optimizer)
        if logging.train_all == 'with_validation':
            train_ind, validate_ind = train_test_split(range(len(train_list_raw)), test_size = 0.05, random_state = logging.sklearn_seed)
            train_data, train_data_list, validate_data, validate_data_list = subset_train_validate(train_data_raw, train_list_raw, train_ind, validate_ind)
            if logging.use_relaxed:
                relaxed_train_data, relaxed_train_data_list, relaxed_validate_data, relaxed_validate_data_list = subset_train_validate(relaxed_train_data_raw, relaxed_train_list_raw, train_ind, validate_ind)
                train_data.extend(relaxed_train_data)
                train_data_list.extend(relaxed_train_data_list)
                validate_data.extend(relaxed_validate_data)
                validate_data_list.extend(relaxed_validate_data_list)
            if logging.use_alphafold:
                af_train_ind, af_validate_ind = train_test_split(range(len(adj_af_train_list)), test_size = 0.05, random_state = logging.sklearn_seed)
                af_train_data, af_train_data_list, af_validate_data, af_validate_data_list = subset_train_validate(af_train_data_raw, adj_af_train_list, af_train_ind, af_validate_ind)
                train_data.extend(af_train_data)
                train_data_list.extend(af_train_data_list)
                validate_data.extend(af_validate_data)
                validate_data_list.extend(af_validate_data_list)
            loss_record.extend(train_class.epoch_training(train_data, validate_data = validate_data))
            evaluation_record.extend(test_model(model, train_data, train_data_list, logging))
            evaluation_record.extend(test_model(model, validate_data, validate_data_list, logging, testType='validate'))
            evaluation_record.extend(test_model(model, test_data, test_list, logging, testType='test'))
        else:
            loss_record.extend(train_class.epoch_training(train_data_raw))
            evaluation_record.extend(test_model(model, train_data_raw, train_list_raw, logging))
        # Attribute gradient
        if logging.gradient_attribute:
            logging.gradient_attribute_weight = model.state_dict()['attribute_layer.weight'].detach().cpu().tolist()
            if logging.gradient_attribute_with_bias:
                logging.gradient_attribute_bias = model.state_dict()['attribute_layer.bias'].detach().cpu().tolist()
        # Saving model
        if not logging.dont_save_model:
            if logging.save_not_as_statedict:
                torch.save(model, os.path.join(logging.directory_output_folder,'model.pt'))
                # model = torch.load(PATH)
                # model.eval()
            else:
                # Saving model by state_dict()
                torch.save(model.state_dict(), os.path.join(logging.directory_output_folder,'model.pt'))
                # model = TheModelClass(*args, **kwargs)
                # model.load_state_dict(torch.load(PATH))
                # model.eval()
    else:
        kf = KFold(n_splits = logging.num_kfold)
        kfold_generator = kf.split(train_data_raw)
        for fold_ind, (train_ind, validate_ind) in enumerate(kfold_generator):
            train_data, train_data_list, validate_data, validate_data_list = subset_train_validate(train_data_raw, train_list_raw, train_ind, validate_ind)
            if logging.use_relaxed:
                relaxed_train_data, relaxed_train_data_list, relaxed_validate_data, relaxed_validate_data_list = subset_train_validate(relaxed_train_data_raw, relaxed_train_list_raw, train_ind, validate_ind)
                train_data.extend(relaxed_train_data)
                train_data_list.extend(relaxed_train_data_list)
                validate_data.extend(relaxed_validate_data)
                validate_data_list.extend(relaxed_validate_data_list)
            if logging.use_alphafold:
                af_train_data, af_train_data_list, af_validate_data, af_validate_data_list = subset_train_validate(af_train_data_raw, adj_af_train_list, af_train_ind, af_validate_ind)
                train_data.extend(af_train_data)
                train_data_list.extend(af_train_data_list)
                validate_data.extend(af_validate_data)
                validate_data_list.extend(af_validate_data_list)
            model = choose_model(logging).to(logging.device)
            optimizer = set_optimizer(model, logging)
            train_class.set_model(model, optimizer)
            loss_record.extend(train_class.epoch_training(train_data, fold_ind, validate_data))
            evaluation_record.extend(test_model(model, train_data, train_data_list, logging, foldInd = fold_ind + 1))
            evaluation_record.extend(test_model(model, validate_data, validate_data_list, logging, testType='validate', foldInd = fold_ind + 1))
            evaluation_record.extend(test_model(model, test_data, test_list, logging, testType='test', foldInd = fold_ind + 1))
            # Attribute gradient
            if logging.gradient_attribute:
                logging.gradient_attribute_weight.extend(model.state_dict()['attribute_layer.weight'].detach().cpu().tolist())
                if logging.gradient_attribute_with_bias:
                    logging.gradient_attribute_bias.extend(model.state_dict()['attribute_layer.bias'].detach().cpu().tolist())
            print()
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return loss_record, evaluation_record