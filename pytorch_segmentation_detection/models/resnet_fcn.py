import torch
import torch.nn as nn
import torchvision


class Resnet18_8s(nn.Module):
    
    # Achieved ~57 on pascal VOC
    
    def __init__(self, num_classes=2):
        
        super(Resnet18_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_32s = torchvision.models.resnet18(fully_conv=True,
                                                   pretrained=True,
                                                   output_stride=32,
                                                   remove_avg_pool_layer=True)
        
        resnet_block_expansion_rate = resnet18_32s.layer1[0].expansion
        
        # Create a linear layer -- we don't need logits in this case
        resnet18_32s.fc = nn.Sequential()
        
        self.resnet18_32s = resnet18_32s
        
        self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_32s.conv1(x)
        x = self.resnet18_32s.bn1(x)
        x = self.resnet18_32s.relu(x)
        x = self.resnet18_32s.maxpool(x)

        x = self.resnet18_32s.layer1(x)
        
        x = self.resnet18_32s.layer2(x)
        logits_8s = self.score_8s(x)
        
        x = self.resnet18_32s.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet18_32s.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
                
        logits_16s += nn.functional.upsample_bilinear(logits_32s,
                                        size=logits_16s_spatial_dim)
        
        logits_8s += nn.functional.upsample_bilinear(logits_16s,
                                        size=logits_8s_spatial_dim)
        
        logits_upsampled = nn.functional.upsample_bilinear(logits_8s,
                                                           size=input_spatial_dim)
        
        return logits_upsampled

    
class Resnet34_8s(nn.Module):
    
    
    def __init__(self, num_classes=2):
        
        super(Resnet34_8s, self).__init__()
        
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_32s = torchvision.models.resnet34(fully_conv=True,
                                                   pretrained=True,
                                                   output_stride=32,
                                                   remove_avg_pool_layer=True)
        
        resnet_block_expansion_rate = resnet34_32s.layer1[0].expansion
        
        # Create a linear layer -- we don't need logits in this case
        resnet34_32s.fc = nn.Sequential()
        
        self.resnet34_32s = resnet34_32s
        
        self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet34_32s.conv1(x)
        x = self.resnet34_32s.bn1(x)
        x = self.resnet34_32s.relu(x)
        x = self.resnet34_32s.maxpool(x)

        x = self.resnet34_32s.layer1(x)
        
        x = self.resnet34_32s.layer2(x)
        logits_8s = self.score_8s(x)
        
        x = self.resnet34_32s.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet34_32s.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
                
        logits_16s += nn.functional.upsample_bilinear(logits_32s,
                                        size=logits_16s_spatial_dim)
        
        logits_8s += nn.functional.upsample_bilinear(logits_16s,
                                        size=logits_8s_spatial_dim)
        
        logits_upsampled = nn.functional.upsample_bilinear(logits_8s,
                                                           size=input_spatial_dim)
        
        return logits_upsampled
    

class Resnet50_8s(nn.Module):
    
    
    def __init__(self, num_classes=2):
        
        super(Resnet50_8s, self).__init__()
        
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.resnet50(fully_conv=True,
                                                   pretrained=True,
                                                   output_stride=32,
                                                   remove_avg_pool_layer=True)
        
        resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion
        
        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()
        
        self.resnet50_32s = resnet50_32s
        
        self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)
        
        x = self.resnet50_32s.layer2(x)
        logits_8s = self.score_8s(x)
        
        x = self.resnet50_32s.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet50_32s.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
                
        logits_16s += nn.functional.upsample_bilinear(logits_32s,
                                        size=logits_16s_spatial_dim)
        
        logits_8s += nn.functional.upsample_bilinear(logits_16s,
                                        size=logits_8s_spatial_dim)
        
        logits_upsampled = nn.functional.upsample_bilinear(logits_8s,
                                                           size=input_spatial_dim)
        
        return logits_upsampled