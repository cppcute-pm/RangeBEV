from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck
import torch.nn as nn
from torch import Tensor
import torch


class ResNet18(ResNet):

    def __init__(self, cfgs, out_dim):
        if 'layers' not in cfgs.keys():
            layers = [2, 2, 2, 2]
        else:
            layers = cfgs['layers']
        if 'block' not in cfgs.keys() or cfgs['block'] == 'BasicBlock':
            block = BasicBlock
        else:
            block = Bottleneck
        super(ResNet18, self).__init__(block=block, 
                                       layers=layers)
        self.f_layer = cfgs.f_layer
        self.c_layer = cfgs.c_layer
        self.out_dim = out_dim
        self.layer_dims = [64, 64, 128, 256, 512]
        self.f_fc = nn.Sequential(
            nn.Conv2d(self.layer_dims[self.f_layer - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_dim),
            )
        self.c_fc = nn.Sequential(
            nn.Conv2d(self.layer_dims[self.c_layer - 1], self.out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_dim),
            )
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.f_layer == 1:
            f_ebds = x
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 1:
            c_ebds = x
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output

        x = self.layer1(x)
        if self.f_layer == 2:
            f_ebds = x
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 2:
            c_ebds = x
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output
        x = self.layer2(x)
        if self.f_layer == 3:
            f_ebds = x
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 3:
            c_ebds = x
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output
        x = self.layer3(x)
        if self.f_layer == 4:
            f_ebds = x
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 4:
            c_ebds = x
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output
        x = self.layer4(x)
        if self.f_layer == 5:
            f_ebds = x
            f_ebds_output = self.f_fc(f_ebds)
        if self.c_layer == 5:
            c_ebds = x
            c_ebds_output = self.c_fc(c_ebds)
            return c_ebds_output, f_ebds_output

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x