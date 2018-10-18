import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch
import math


class PSP_head(nn.Module):
    
    def __init__(self, in_channels):
        
        super(PSP_head, self).__init__()
        
        out_channels = int( in_channels / 4 )
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.fusion_bottleneck = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(out_channels),
                                               nn.ReLU(True),
                                               nn.Dropout2d(0.1, False))
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):

        fcn_features_spatial_dim = x.size()[2:]

        pooled_1 = nn.functional.adaptive_avg_pool2d(x, 1)
        pooled_1 = self.conv1(pooled_1)
        pooled_1 = nn.functional.upsample_bilinear(pooled_1, size=fcn_features_spatial_dim)

        pooled_2 = nn.functional.adaptive_avg_pool2d(x, 2)
        pooled_2 = self.conv2(pooled_2)
        pooled_2 = nn.functional.upsample_bilinear(pooled_2, size=fcn_features_spatial_dim)

        pooled_3 = nn.functional.adaptive_avg_pool2d(x, 3)
        pooled_3 = self.conv3(pooled_3)
        pooled_3 = nn.functional.upsample_bilinear(pooled_3, size=fcn_features_spatial_dim)

        pooled_4 = nn.functional.adaptive_avg_pool2d(x, 6)
        pooled_4 = self.conv4(pooled_4)
        pooled_4 = nn.functional.upsample_bilinear(pooled_4, size=fcn_features_spatial_dim)

        x = torch.cat([x, pooled_1, pooled_2, pooled_3, pooled_4],
                       dim=1)

        x = self.fusion_bottleneck(x)

        return x

        
class Resnet50_8s_psp(nn.Module):
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_8s_psp, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet50_8s = models.resnet50(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        
        self.psp_head = PSP_head(resnet50_8s.inplanes)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes // 4, num_classes, 1)
        
        self.resnet50_8s = resnet50_8s
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s.conv1(x)
        x = self.resnet50_8s.bn1(x)
        x = self.resnet50_8s.relu(x)
        x = self.resnet50_8s.maxpool(x)

        x = self.resnet50_8s.layer1(x)
        x = self.resnet50_8s.layer2(x)
        x = self.resnet50_8s.layer3(x)
        x = self.resnet50_8s.layer4(x)
        
        x = self.psp_head(x)
        
        x = self.resnet50_8s.fc(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x