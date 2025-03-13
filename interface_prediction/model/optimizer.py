import torch

def set_optimizer(modelBuild, logging):
    '''
    This function for setting optimizer.
    '''
    optimizer_method = logging.optimizer_method.lower()
    assert optimizer_method in ['adam',
                                'momentum',
                                'sgd'], f'{logging.optimizer_method} is not defined.'
    if optimizer_method == 'adam':
        return torch.optim.Adam(modelBuild.parameters(), 
                                lr = logging.learning_rate, 
                                weight_decay = logging.weight_decay)
    elif optimizer_method == 'momentum':
        return torch.optim.SGD(modelBuild.parameters(), 
                               lr = logging.learning_rate, 
                               weight_decay = logging.weight_decay, 
                               momentum = logging.momentum)
    elif optimizer_method == 'sgd':
        return torch.optim.SGD(modelBuild.parameters(), 
                               lr = logging.learning_rate, 
                               weight_decay = logging.weight_decay) 