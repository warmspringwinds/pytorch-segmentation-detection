import numpy as np
import torch.nn as nn
import torchvision.models as models


def adjust_input_image_size_for_proper_feature_alignment(input_img_batch, output_stride=8):
    """Resizes the input image to allow proper feature alignment during the
    forward propagation.

    Resizes the input image to a closest multiple of `output_stride` + 1.
    This allows the proper alignment of features.
    To get more details, read here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159

    Parameters
    ----------
    input_img_batch : torch.Tensor
        Tensor containing a single input image of size (1, 3, h, w)

    output_stride : int
        Output stride of the network where the input image batch
        will be fed.

    Returns
    -------
    input_img_batch_new_size : torch.Tensor
        Resized input image batch tensor
    """

    input_spatial_dims = np.asarray( input_img_batch.shape[2:], dtype=np.float )

    # Comments about proper alignment can be found here
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159
    new_spatial_dims = np.ceil(input_spatial_dims / output_stride).astype(np.int) * output_stride + 1

    # Converting the numpy to list, torch.nn.functional.upsample_bilinear accepts
    # size in the list representation.
    new_spatial_dims = list(new_spatial_dims)

    input_img_batch_new_size = nn.functional.upsample_bilinear(input=input_img_batch,
                                                               size=new_spatial_dims)

    return input_img_batch_new_size



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
        
    def forward(self, x, feature_alignment=False):
        
        input_spatial_dim = x.size()[2:]
        
        if feature_alignment:
            
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)
        
        x = self.resnet18_8s(x)
        
        x = nn.functional.upsample(x, size=input_spatial_dim, mode='bilinear', align_corners=True)
        
        #x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)#, align_corners=False)
        
        return x

    
class Resnet18_16s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet18_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet18_16s = models.resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=16,
                                      remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_16s.fc = nn.Conv2d(resnet18_16s.inplanes, num_classes, 1)
        
        self.resnet18_16s = resnet18_16s
        
        self._normal_initialization(self.resnet18_16s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_16s(x)
        
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
    

    
class Resnet34_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet34_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_32s = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_32s.fc = nn.Conv2d(resnet34_32s.inplanes, num_classes, 1)
        
        self.resnet34_32s = resnet34_32s
        
        self._normal_initialization(self.resnet34_32s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet34_32s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

    
class Resnet34_16s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet34_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_16s = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=16,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_16s.fc = nn.Conv2d(resnet34_16s.inplanes, num_classes, 1)
        
        self.resnet34_16s = resnet34_16s
        
        self._normal_initialization(self.resnet34_16s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet34_16s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x


class Resnet34_8s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet34_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)
        
        self.resnet34_8s = resnet34_8s
        
        self._normal_initialization(self.resnet34_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x, feature_alignment=False):
        
        input_spatial_dim = x.size()[2:]
        
        if feature_alignment:
            
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)
        
        x = self.resnet34_8s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
    
class Resnet50_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = models.resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_32s.fc = nn.Conv2d(resnet50_32s.inplanes, num_classes, 1)
        
        self.resnet50_32s = resnet50_32s
        
        self._normal_initialization(self.resnet50_32s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_32s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

    
class Resnet50_16s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet50_8s = models.resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=16,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, num_classes, 1)
        
        self.resnet50_8s = resnet50_8s
        
        self._normal_initialization(self.resnet50_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

class Resnet50_8s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_8s = models.resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, num_classes, 1)
        
        self.resnet50_8s = resnet50_8s
        
        self._normal_initialization(self.resnet50_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

    

class Resnet9_8s(nn.Module):
    
    # Gets ~ 46 MIOU on Pascal Voc
    
    def __init__(self, num_classes=1000):
        
        super(Resnet9_8s, self).__init__()
        
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
        
        x = self.resnet18_8s.conv1(x)
        x = self.resnet18_8s.bn1(x)
        x = self.resnet18_8s.relu(x)
        x = self.resnet18_8s.maxpool(x)

        x = self.resnet18_8s.layer1[0](x)
        x = self.resnet18_8s.layer2[0](x)
        x = self.resnet18_8s.layer3[0](x)
        x = self.resnet18_8s.layer4[0](x)
        
        x = self.resnet18_8s.fc(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x