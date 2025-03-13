import torch
from torch import nn
import torch.nn.functional as F

class CustomMSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: torch.tensor, target: torch.Tensor) -> torch.tensor:
        return F.mse_loss(input.reshape(-1), target.float())
    
def get_loss_function(loss_function, 
                      cross_entropy_weight):
    if loss_function == 'cross_entropy': 
        if cross_entropy_weight:
            return nn.CrossEntropyLoss(weight=torch.tensor(cross_entropy_weight))
        else:
            return nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        return CustomMSELoss()
    
