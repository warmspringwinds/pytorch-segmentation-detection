import torch.nn as nn
import torchvision.models as models


class FCN_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(FCN_32s, self).__init__()
        
        vgg16 = models.vgg16(pretrained=True,
                             fully_conv=True,
                             num_classes=num_classes)
        
        self.features = vgg16.features
        self.classifier = vgg16.classifier
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.features(x)
        x = self.classifier(x)
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
