from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResNetFPNv2(ResNet):

    def __init__(self, cfgs, out_dim):
        super(ResNetFPNv2, self).__init__(block=BasicBlock, 
                                       layers=[2, 2, 2, 2])
        self.f_layer = cfgs.f_layer
        self.c_layer = cfgs.c_layer
        self.out_dim = out_dim
        self.layer_dims = [64, 64, 128, 256, 512, 512, 256, 128]
        self.f_fc = nn.Sequential(
            nn.Conv2d(self.layer_dims[self.f_layer - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_dim),
            )
        self.c_fc = nn.Sequential(
            nn.Conv2d(self.layer_dims[self.c_layer - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_dim),
            )
        
        # 3. FPN upsample
        self.layer4_outconv = conv1x1(self.layer_dims[4], self.layer_dims[4])
        self.layer3_outconv = conv1x1(self.layer_dims[3], self.layer_dims[4])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(self.layer_dims[4], self.layer_dims[4]),
            nn.BatchNorm2d(self.layer_dims[4]),
            nn.LeakyReLU(),
            conv3x3(self.layer_dims[4], self.layer_dims[3]),
        )

        self.layer2_outconv = conv1x1(self.layer_dims[2], self.layer_dims[3])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(self.layer_dims[3], self.layer_dims[3]),
            nn.BatchNorm2d(self.layer_dims[3]),
            nn.LeakyReLU(),
            conv3x3(self.layer_dims[3], self.layer_dims[2]),
        )
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        if self.f_layer == 1:
            f_ebds = x0
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 1:
            c_ebds = x0
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output

        x1 = self.layer1(x0)
        if self.f_layer == 2:
            f_ebds = x1
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 2:
            c_ebds = x1
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output
        x2 = self.layer2(x1)
        if self.f_layer == 3:
            f_ebds = x2
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 3:
            c_ebds = x2
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output
        x3 = self.layer3(x2)
        if self.f_layer == 4:
            f_ebds = x3
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 4:
            c_ebds = x3
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output
        x4 = self.layer4(x3)
        if self.f_layer == 5:
            f_ebds = x4
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 5:
            c_ebds = x4
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output
        
        # FPN
        x4_out = self.layer4_outconv(x4)
        if self.f_layer == 6:
            f_ebds = x4_out
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 6:
            c_ebds = x4_out
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output
        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out+x4_out_2x)
        if self.f_layer == 7:
            f_ebds = x3_out
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 7:
            c_ebds = x3_out
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)
        if self.f_layer == 8:
            f_ebds = x2_out
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 8:
            c_ebds = x2_out
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output