import torch
import numpy as np
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    "3x3 convolution with padding"
    
    kernel_size = np.asarray((3, 3))
    
    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    
    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2
    
    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)
    
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)


class ASPP(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels_per_branch=256,
                 branch_dilations=(6, 12, 18)):
        
        super(ASPP, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels_per_branch,
                                  kernel_size=1,
                                  bias=False)
        
        self.conv_1x1_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
        self.conv_3x3_first = conv3x3(in_channels, out_channels_per_branch, dilation=branch_dilations[0])
        self.conv_3x3_first_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
        
        self.conv_3x3_second = conv3x3(in_channels, out_channels_per_branch, dilation=branch_dilations[1])
        self.conv_3x3_second_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
        
        self.conv_3x3_third = conv3x3(in_channels, out_channels_per_branch, dilation=branch_dilations[2])
        self.conv_3x3_third_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
        self.conv_1x1_pool = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels_per_branch,
                                       kernel_size=1,
                                       bias=False)
        self.conv_1x1_pool_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
        self.conv_1x1_final = nn.Conv2d(in_channels=out_channels_per_branch * 5,
                                        out_channels=out_channels_per_branch,
                                        kernel_size=1,
                                        bias=False)
        self.conv_1x1_final_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)
        
    
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        conv_1x1_branch = self.relu(self.conv_1x1_bn(self.conv_1x1(x)))
        conv_3x3_first_branch = self.relu(self.conv_3x3_first_bn(self.conv_3x3_first(x)))
        conv_3x3_second_branch = self.relu(self.conv_3x3_second_bn(self.conv_3x3_second(x)))
        conv_3x3_third_branch = self.relu(self.conv_3x3_third_bn(self.conv_3x3_third(x)))
        
        global_pool_branch = self.relu(self.conv_1x1_pool_bn(self.conv_1x1_pool(nn.functional.adaptive_avg_pool2d(x, 1))))
        global_pool_branch = nn.functional.upsample_bilinear(input=global_pool_branch,
                                                             size=input_spatial_dim)
        
        features_concatenated = torch.cat([conv_1x1_branch,
                                           conv_3x3_first_branch,
                                           conv_3x3_second_branch,
                                           conv_3x3_third_branch,
                                           global_pool_branch],
                                          dim=1)
        
        features_fused = self.relu(self.conv_1x1_final_bn(self.conv_1x1_final(features_concatenated)))
        
        return features_fused