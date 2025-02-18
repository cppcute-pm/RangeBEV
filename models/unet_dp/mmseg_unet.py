from mmseg.models.decode_heads import FCNHead
from mmseg.models.backbones import UNet
import torch.nn as nn
import torch

# class FCNHead_v2(FCNHead):

#     def _forward_feature(self, inputs):
#         x = self._transform_inputs(inputs)
#         feats = self.convs(x)
#         if self.concat_input:
#             feats = self.conv_cat(torch.cat([x, feats], dim=1))
#         return feats

#     def forward(self, inputs):
#         output = self._forward_feature(inputs)
#         return output

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class UNet_MMSeg(nn.Module):
    def __init__(self, cfgs, out_dim):
        super(UNet_MMSeg, self).__init__()
        norm_cfg = dict(type='BN', requires_grad=True)
        self.backbone = UNet(in_channels=3, 
                             base_channels=64, 
                             num_stages=5, 
                             strides=(1, 1, 1, 1, 1),
                             enc_num_convs=(2, 2, 2, 2, 2),
                             dec_num_convs=(2, 2, 2, 2),
                             downsamples=(True, True, True, True),
                             enc_dilations=(1, 1, 1, 1, 1),
                             dec_dilations=(1, 1, 1, 1),
                             with_cp=False,
                             conv_cfg=None,
                             norm_cfg=norm_cfg,
                             act_cfg=dict(type='ReLU'),
                             upsample_cfg=dict(type='InterpConv'),
                             norm_eval=False)
        self.f_layer = cfgs.f_layer
        self.c_layer = cfgs.c_layer
        self.all_channels = [1024, 512, 256, 128, 64]
        self.c_fc = nn.Sequential(
            conv1x1(self.all_channels[self.c_layer - 1], out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.f_fc = nn.Sequential(
            conv1x1(self.all_channels[self.f_layer - 1], out_dim),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x1 = self.backbone(x)
        f_feats = self.f_fc(x1[self.f_layer - 1])
        c_feats = self.c_fc(x1[self.c_layer - 1])
        return f_feats, c_feats