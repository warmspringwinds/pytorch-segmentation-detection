import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch


class Resnet50_8s_psp(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_8s_psp, self).__init__()
        
        self.is_network_split_over_two_gpus = False
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet50_8s = models.resnet50(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes * 2, num_classes, 1)
        
        self.resnet50_8s = resnet50_8s
        
        self.reduction_pooled_1 = nn.Conv2d(resnet50_8s.inplanes, 512, 1)
        self.reduction_pooled_2 = nn.Conv2d(resnet50_8s.inplanes, 512, 1)
        self.reduction_pooled_3 = nn.Conv2d(resnet50_8s.inplanes, 512, 1)
        self.reduction_pooled_4 = nn.Conv2d(resnet50_8s.inplanes, 512, 1)
        
        
    def split_network_over_two_gpus(self, gpus=[0, 1]):
        
        self.resnet50_8s.conv1.cuda(gpus[0])
        self.resnet50_8s.bn1.cuda(gpus[0])
        self.resnet50_8s.relu.cuda(gpus[0])
        self.resnet50_8s.maxpool.cuda(gpus[0])

        self.resnet50_8s.layer1.cuda(gpus[0])
        self.resnet50_8s.layer2.cuda(gpus[0])
        self.resnet50_8s.layer3.cuda(gpus[0])
        
        
        self.resnet50_8s.layer4.cuda(gpus[1])
        self.reduction_pooled_1.cuda(gpus[1])
        self.reduction_pooled_2.cuda(gpus[1])
        self.reduction_pooled_3.cuda(gpus[1])
        self.reduction_pooled_4.cuda(gpus[1])
        
        self.resnet50_8s.fc.cuda(gpus[1])
        
        self.is_network_split_over_two_gpus = True
        self.split_gpus = gpus
    
    def cuda(self, device=None):
        
        super(Resnet50_8s_psp, self).cuda(device)
        
        self.is_network_split_over_two_gpus = False
    
    def cpu(self):
        
        super(Resnet50_8s_psp, self).cpu()
        
        self.is_network_split_over_two_gpus = False
        
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s.conv1(x)
        x = self.resnet50_8s.bn1(x)
        x = self.resnet50_8s.relu(x)
        x = self.resnet50_8s.maxpool(x)

        x = self.resnet50_8s.layer1(x)
        x = self.resnet50_8s.layer2(x)
        x = self.resnet50_8s.layer3(x)
        
        if self.is_network_split_over_two_gpus:
            
            x = x.cuda(self.split_gpus[1])
        
        x = self.resnet50_8s.layer4(x)
        
        fcn_features_spatial_dim = x.size()[2:]
        
        pooled_1 = nn.functional.adaptive_avg_pool2d(x, 1)
        pooled_1 = self.reduction_pooled_1(pooled_1)
        pooled_1 = nn.functional.upsample_bilinear(pooled_1, size=fcn_features_spatial_dim)
        
        pooled_2 = nn.functional.adaptive_avg_pool2d(x, 2)
        pooled_2 = self.reduction_pooled_1(pooled_2)
        pooled_2 = nn.functional.upsample_bilinear(pooled_1, size=fcn_features_spatial_dim)
        
        pooled_3 = nn.functional.adaptive_avg_pool2d(x, 3)
        pooled_3 = self.reduction_pooled_1(pooled_3)
        pooled_3 = nn.functional.upsample_bilinear(pooled_1, size=fcn_features_spatial_dim)
        
        pooled_4 = nn.functional.adaptive_avg_pool2d(x, 6)
        pooled_4 = self.reduction_pooled_1(pooled_4)
        pooled_4 = nn.functional.upsample_bilinear(pooled_1, size=fcn_features_spatial_dim)
        
        x = torch.cat([x, pooled_1, pooled_2, pooled_3, pooled_4],
                      dim=1)
        
        x = self.resnet50_8s.fc(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x