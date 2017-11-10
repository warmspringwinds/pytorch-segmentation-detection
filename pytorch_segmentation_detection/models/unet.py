import torch
import torch.nn as nn



def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     padding=1,
                     bias=True)


class UnetDownBlock(nn.Module):
    """ Downsampling block of Unet.
    
        The whole architercture of Unet has a one common pattern: a block
        that spatially downsamples the input followed by two layers of 3x3 convolutions that
        has 'inplanes' number of input planes and 'planes' number of channels.
    
    """
    
    def __init__(self, inplanes, planes, predownsample_block):
        
        super(UnetDownBlock, self).__init__()
        
        self.predownsample_block = predownsample_block
        self.conv1 = conv3x3(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        
    def forward(self, x):
        
        x = self.predownsample_block(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        return x


class UnetUpBlock(nn.Module):
    """ Upsampling block of Unet.
    
        The whole architercture of Unet has a one common pattern: a block
        that has two layers of 3x3 convolutions that
        has 'inplanes' number of input planes and 'planes' number of channels,
        followed by 'postupsample_block' which increases the spatial resolution
    
    """
    
    def __init__(self, inplanes, planes, postupsample_block=None):
        
        super(UnetUpBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        if postupsample_block is None:
        
            self.postupsample_block = torch.nn.ConvTranspose2d(in_channels=planes,
                                                               out_channels=planes/2,
                                                               kernel_size=2,
                                                               stride=2)
        else:
            
            self.postupsample_block = postupsample_block
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.postupsample_block(x)
        
        return x
        

class Unet(nn.Module):
    """Unet network. ~297 ms on hd image."""
    
    def __init__(self, num_classes=2):
        
        super(Unet, self).__init__()
        
        self.predownsample_block = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.identity_block = nn.Sequential()
        
        self.block1 = UnetDownBlock(
                                    predownsample_block=self.identity_block,
                                    inplanes=3,
                                    planes=64,
                                    )
            
        self.block2_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=64,
                                         planes=128,
                                         )
        
        self.block3_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=128,
                                         planes=256
                                        )
        
        self.block4_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=256,
                                         planes=512
                                        )
        
        self.block5_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=512,
                                         planes=1024
                                         )
        
        self.block1_up = torch.nn.ConvTranspose2d(in_channels=1024,
                                                  out_channels=512,
                                                  kernel_size=2,
                                                  stride=2)
        
        self.block2_up = UnetUpBlock(
                                    inplanes=1024,
                                    planes=512
                                    )
        
        self.block3_up = UnetUpBlock(
                                    inplanes=512,
                                    planes=256
                                    )
        
        self.block4_up = UnetUpBlock(
                                    inplanes=256,
                                    planes=128
                                    )
        
        self.block5 = UnetUpBlock(
                                  inplanes=128,
                                  planes=64,
                                  postupsample_block=self.identity_block
                                 )
        
        self.logit_conv = nn.Conv2d(64,
                                    num_classes,
                                    kernel_size=1)
        
    
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        # Left part of the U figure in the Unet paper
        features_1s_down = self.block1(x)
        features_2s_down = self.block2_down(features_1s_down)
        features_4s_down = self.block3_down(features_2s_down)
        features_8s_down = self.block4_down(features_4s_down)
        
        # Bottom part of the U figure in the Unet paper
        features_16s = self.block5_down(features_8s_down)
        
        # Right part of the U figure in the Unet paper
        features_8s_up = self.block1_up(features_16s)
        features_8s_up = torch.cat([features_8s_down, features_8s_up], dim=1)
        
        features_4s_up = self.block2_up(features_8s_up)
        features_4s_up = torch.cat([features_4s_down, features_4s_up], dim=1)
        
        features_2s_up = self.block3_up(features_4s_up)
        features_2s_up = torch.cat([features_2s_down, features_2s_up], dim=1)
        
        features_1s_up = self.block4_up(features_2s_up)
        features_1s_up = torch.cat([features_1s_down, features_1s_up], dim=1)
        
        features_final = self.block5(features_1s_up)
        
        logits = self.logit_conv(features_final)
            
        return logits