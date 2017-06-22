import torch.nn as nn
import torchvision.models as models


class Resnet101_8s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet101_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet101_8s = models.resnet101(fully_conv=True,
                                        pretrained=True,
                                        output_stride=8,
                                        remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet101_8s.fc = nn.Conv2d(resnet101_8s.inplanes, num_classes, 1)
        
        self.resnet101_8s = resnet101_8s
        
        self._normal_initialization(self.resnet101_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet101_8s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
    

    
class Resnet18_8s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet18_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = models.resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Conv2d(resnet18_8s.inplanes, num_classes, 1)
        
        self.resnet18_8s = resnet18_8s
        
        self._normal_initialization(self.resnet18_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_8s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
    

class Resnet18_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet18_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_32s = models.resnet18(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_32s.fc = nn.Conv2d(resnet18_32s.inplanes, num_classes, 1)
        
        self.resnet18_32s = resnet18_32s
        
        self._normal_initialization(self.resnet18_32s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_32s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x