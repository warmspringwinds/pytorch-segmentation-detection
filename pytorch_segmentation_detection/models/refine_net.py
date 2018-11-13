class RefineNet(nn.Module):
    
    def __init__(self, num_classes=2):
        """http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0632.pdf
        
        is temporary placed here
        """
        
        super(RefineNet, self).__init__()
        
        
        resnet50_8s = torchvision.models.resnet18(fully_conv=True,
                                                   pretrained=True,
                                                   output_stride=32,
                                                   remove_avg_pool_layer=True)
        
        
        resnet_block_expansion_rate = resnet50_8s.layer1[0].expansion
        
        self.logit_conv = nn.Conv2d(512,
                                    num_classes,
                                    kernel_size=1)
        
        self.layer_4_refine_left = BasicBlock(inplanes=512, planes=512)
        self.layer_4_refine_right = BasicBlock(inplanes=512, planes=512)

        self.layer_3_downsample = create_downsample_path(256, 512)
        self.layer_3_refine_left = BasicBlock(inplanes=256, planes=512, downsample=self.layer_3_downsample)
        self.layer_3_refine_right = BasicBlock(inplanes=512, planes=512)

        self.layer_2_downsample = create_downsample_path(128, 512)
        self.layer_2_refine_left = BasicBlock(inplanes=128, planes=512, downsample=self.layer_2_downsample)
        self.layer_2_refine_right = BasicBlock(inplanes=512, planes=512)

        self.layer_1_downsample = create_downsample_path(64, 512)
        self.layer_1_refine_left = BasicBlock(inplanes=64, planes=512, downsample=self.layer_1_downsample)
        self.layer_1_refine_right = BasicBlock(inplanes=512, planes=512)
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.resnet50_8s = resnet50_8s
        
        
    def forward(self, x):
        
        # Get spatiall sizes
        input_height = x.size(2)
        input_width = x.size(3)
        
        # We don't gate the first stage of resnet
        x = self.resnet50_8s.conv1(x)
        x = self.resnet50_8s.bn1(x)
        x = self.resnet50_8s.relu(x)
        x = self.resnet50_8s.maxpool(x)
        
        layer1_output = self.resnet50_8s.layer1(x)
        layer2_output = self.resnet50_8s.layer2(layer1_output)
        layer3_output = self.resnet50_8s.layer3(layer2_output)
        layer4_output = self.resnet50_8s.layer4(layer3_output)
        
        global_pool = nn.functional.adaptive_avg_pool2d(layer4_output, 1)
        global_pool = nn.functional.upsample_bilinear(global_pool, size=layer4_output.size()[2:])
        layer_4_refined = self.layer_4_refine_right( self.layer_4_refine_left(layer4_output) + global_pool )
        
        
        layer_4_refined = nn.functional.upsample_bilinear(layer_4_refined, size=layer3_output.size()[2:])
        layer_3_refined = self.layer_3_refine_right( self.layer_3_refine_left(layer3_output) + layer_4_refined )
        
        layer_3_refined = nn.functional.upsample_bilinear(layer_3_refined, size=layer2_output.size()[2:])
        layer_2_refined = self.layer_2_refine_right( self.layer_2_refine_left(layer2_output) + layer_3_refined )
        
        layer_2_refined = nn.functional.upsample_bilinear(layer_2_refined, size=layer1_output.size()[2:])
        layer_1_refined = self.layer_1_refine_right( self.layer_1_refine_left(layer1_output) + layer_2_refined )
        
        logits = self.logit_conv(layer_1_refined)
        
        logits_upsampled = nn.functional.upsample_bilinear(logits,
                                                           size=(input_height, input_width))
        
        
        return logits_upsampled