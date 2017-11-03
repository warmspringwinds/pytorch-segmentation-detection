import torch
import torch.nn as nn


class GlobalAvgPool2d(nn.Module):
    """ Global Average pooling over last two spatial dimensions. """
    
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, input):
        return input.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)

    def __repr__(self):
        return self.__class__.__name__ + '( )'