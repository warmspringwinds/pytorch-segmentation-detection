import torch
import torch.nn as nn
import torch.nn.functional as F
from .transforms import convert_labels_to_one_hot_encoding

# TODO: version of pytorch for cuda 7.5 doesn't have the latest features like
# reduce=False argument -- update cuda on the machine and update the code

# TODO: update the class to inherit the nn.Weighted loss with all the additional
# arguments

class FocalLoss(nn.Module):
    """Focal loss puts more weight on more complicated examples."""
   
    def __init__(self, gamma=1):
        
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, flatten_logits, flatten_targets):
        
        flatten_targets = flatten_targets.data
        
        number_of_classes = flatten_logits.size(1)
        
        flatten_targets_one_hot = convert_labels_to_one_hot_encoding(flatten_targets, number_of_classes)

        all_class_probabilities = F.softmax(flatten_logits)

        probabilities_of_target_classes = all_class_probabilities[flatten_targets_one_hot]

        elementwise_loss =  - (1 - probabilities_of_target_classes).pow(self.gamma) * torch.log(probabilities_of_target_classes)
        
        return elementwise_loss.sum()