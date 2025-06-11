import torch
from torch import nn
import torch.nn.functional as F

class CustomMSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: torch.tensor, target: torch.Tensor) -> torch.tensor:
        return F.mse_loss(input.reshape(-1), target.float())
    
class HierarchicalCELoss(nn.Module):
    def __init__(self, cross_entropy_weight, device):
        super().__init__()
        self.reachability_matrix = torch.tensor([[1,1,1],
                                                 [0,1,0],
                                                 [1,1,1]]).float().to(device)
        self.cross_entropy_weight = torch.tensor(cross_entropy_weight).to(device)
        self.device = device
        
    def forward(self, logits, targets):
        """
        Hierarchical Cross-Entropy loss
        
        Args:
            logits: Raw model predictions (batch_size, num_classes)
            targets: Ground truth class indices (batch_size)
            reachability_matrix: Matrix encoding hierarchical relationships (num_classes, num_classes)
            weight: Optional class weights
        
        Returns:
            Hierarchical Cross-Entropy loss value
        https://github.com/microsoft/hce-classification
        """
        # Convert logits to probabilities using softmax
        cell_type_probs = torch.softmax(logits, dim=-1)
        
        # Propagate probabilities through the hierarchy using the reachability matrix
        cell_type_probs = torch.matmul(cell_type_probs, self.reachability_matrix.T)
        
        # Apply log transform (with numerical stability term) for NLL loss calculation
        cell_type_probs = torch.log(
            cell_type_probs + torch.tensor(1e-6, device=self.device)
        )
        
        # Calculate negative log-likelihood loss with optional class weights
        hce_loss = F.nll_loss(cell_type_probs, targets, weight=self.cross_entropy_weight).to(self.device)
        return hce_loss
    
def get_loss_function(loss_function, 
                      cross_entropy_weight,
                      device):
    if loss_function == 'cross_entropy': 
        if cross_entropy_weight:
            return nn.CrossEntropyLoss(weight=torch.tensor(cross_entropy_weight))
        else:
            return nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        return CustomMSELoss()
    elif loss_function == 'hce':
        return HierarchicalCELoss(cross_entropy_weight, device)
    
