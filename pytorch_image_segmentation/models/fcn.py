import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchvision.models.vgg import model_urls


class FCN_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(FCN_32s, self).__init__()
        
        vgg16 = models.vgg16(pretrained=True,
                             fully_conv=True)
        
        # Copy all the features
        self.features = vgg16.features
        
        # Remove the last classification 1x1 convolution
        fully_conv = list(vgg16.classifier.children())
        fully_conv = fully_conv[:-1]
        self.fully_conv = nn.Sequential(*fully_conv)
        
        # Get a new 1x1 convolution and randomly initialize
        score_32s = nn.Conv2d(4096, num_classes, 1)
        self._normal_initialization(score_32s)
        self.score_32s = score_32s
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.features(x)
        x = self.fully_conv(x)
        x = self.score_32s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x