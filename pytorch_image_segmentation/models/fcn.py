import torch.nn as nn
import torchvision.models as models


class FCN_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(FCN_32s, self).__init__()
        
        # Load the model with convolutionalized
        # fully connected layers
        vgg16 = models.vgg16(pretrained=True,
                             fully_conv=True)
        
        # Copy all the feature layers as is
        self.features = vgg16.features
        
        # TODO: check if Dropout works correctly for
        # fully convolutional mode
        
        # Remove the last classification 1x1 convolution
        # because it comes from imagenet 1000 class classification.
        # We will perform classification on different classes
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