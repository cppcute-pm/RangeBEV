from mmseg.models.backbones import ResNet 
from mmseg.models.backbones.resnet import BasicBlock, Bottleneck
from mmengine.model import BaseModule
from torch import nn
from .res_layer_myself import ResLayer_m1
from mmcv.cnn import build_upsample_layer
import torch
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class Bottleneck_exp_rate_1(Bottleneck):
    expansion = 1

class Bottleneck_exp_rate_2(Bottleneck):
    expansion = 2

class BasicBlock_exp_rate_2(BasicBlock):
    expansion = 2

class BasicBlock_exp_rate_4(BasicBlock):
    expansion = 4

class ResNet_encoder(ResNet):
    
    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs = [x]
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            outs.append(x)
        return tuple(outs)

class ResNet_decoder(ResNet):

    def __init__(self,                  
                 channels_list=[2048, 1024, 512, 256, 64],
                 stage_blocks=[3, 6, 4, 3],
                 block=Bottleneck,
                 strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 contract_dilation=True,
                 upsample_cfg=dict(mode='InterpConv'),
                 last_layer_num_blocks=1):
        BaseModule.__init__(self, None)
        self.pretrained = False
        self.zero_init_residual = True
        block_init_cfg = None
        self.block = block
        self.stage_blocks = stage_blocks
        self.init_cfg = [
            dict(type='Kaiming', layer='Conv2d'),
            dict(
                type='Constant',
                val=1,
                layer=['_BatchNorm', 'GroupNorm'])
        ]
        if self.zero_init_residual:
            if block is BasicBlock or block is BasicBlock_exp_rate_2 or block is BasicBlock_exp_rate_4:
                block_init_cfg = dict(
                    type='Constant',
                    val=0,
                    override=dict(name='norm2'))
            elif block is Bottleneck or block is Bottleneck_exp_rate_1 or block is Bottleneck_exp_rate_2:
                block_init_cfg = dict(
                    type='Constant',
                    val=0,
                    override=dict(name='norm3'))
        
        self.channels_list = channels_list
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == len(stage_blocks)
        self.style = style
        self.avg_down = False
        self.frozen_stages = -1
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = False
        self.norm_eval = norm_eval
        self.dcn = None
        self.plugins = None
        self.multi_grid = None
        self.contract_dilation = contract_dilation

        self.res_layers = []
        self.upsample_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = None
            if self.plugins is not None:
                stage_plugins = self.make_stage_plugins(self.plugins, i)
            else:
                stage_plugins = None
            # multi grid is applied to last layer only
            stage_multi_grid = self.multi_grid if i == len(
                self.stage_blocks) - 1 else None
        
            upsampler_layer=build_upsample_layer(
                            cfg=upsample_cfg,
                            in_channels=self.channels_list[i],
                            out_channels=self.channels_list[i+1],
                            with_cp=self.with_cp,
                            norm_cfg=norm_cfg,
                            act_cfg=dict(type='ReLU'))
            upsample_layer_name = f'upsample_layer{i+1}'
            self.add_module(upsample_layer_name, upsampler_layer)
            self.upsample_layers.append(upsample_layer_name)
            inplanes = 2 * channels_list[i+1]

            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=channels_list[i+1]//block.expansion,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=self.with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                avg_down=self.avg_down,
                dcn=dcn,
                plugins=stage_plugins,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation,
                init_cfg=block_init_cfg)
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        inplanes = 2 * channels_list[-1]
        self.last_layer = self.make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=channels_list[-1]//block.expansion,
                num_blocks=last_layer_num_blocks,
                stride=1,
                dilation=1,
                style=self.style,
                with_cp=self.with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                avg_down=self.avg_down,
                dcn=dcn,
                plugins=stage_plugins,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation,
                init_cfg=block_init_cfg)
    
    # def make_res_layer(self, **kwargs):
    #     return ResLayer_m1(**kwargs)
    
    def forward(self, x):
        """Forward function."""
        outs = []
        curr_x = x[-1]
        for i, layer_name in enumerate(self.res_layers):
            upsample_layer = getattr(self, self.upsample_layers[i])
            curr_x = upsample_layer(curr_x)
            curr_x = torch.cat([x[-(i+2)], curr_x], dim=1)
            res_layer = getattr(self, layer_name)
            curr_x = res_layer(curr_x)
            outs.append(curr_x)
        curr_x = torch.cat([x[0], curr_x], dim=1)
        curr_x = self.last_layer(curr_x)
        outs.append(curr_x)
        return tuple(outs)

class ResNet_encoder_v2(ResNet):

    def __init__(self,                  
                 channels_list=[64, 256, 512, 1024, 2048],
                 stage_blocks=[3, 6, 4, 3],
                 block=Bottleneck,
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 contract_dilation=True):
        BaseModule.__init__(self, None)
        self.pretrained = False
        self.zero_init_residual = True
        block_init_cfg = None
        self.block = block
        self.stage_blocks = stage_blocks
        self.init_cfg = [
            dict(type='Kaiming', layer='Conv2d'),
            dict(
                type='Constant',
                val=1,
                layer=['_BatchNorm', 'GroupNorm'])
        ]
        if self.zero_init_residual:
            if block is BasicBlock or block is BasicBlock_exp_rate_2 or block is BasicBlock_exp_rate_4:
                block_init_cfg = dict(
                    type='Constant',
                    val=0,
                    override=dict(name='norm2'))
            elif block is Bottleneck or block is Bottleneck_exp_rate_1 or block is Bottleneck_exp_rate_2:
                block_init_cfg = dict(
                    type='Constant',
                    val=0,
                    override=dict(name='norm3'))
        
        self.channels_list = channels_list
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == len(stage_blocks)
        self.style = style
        self.avg_down = False
        self.frozen_stages = -1
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = False
        self.norm_eval = norm_eval
        self.dcn = None
        self.plugins = None
        self.multi_grid = None
        self.contract_dilation = contract_dilation

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = None
            if self.plugins is not None:
                stage_plugins = self.make_stage_plugins(self.plugins, i)
            else:
                stage_plugins = None
            # multi grid is applied to last layer only
            stage_multi_grid = self.multi_grid if i == len(
                self.stage_blocks) - 1 else None
            if i != 0:
                inplanes = 2 * channels_list[i]
            else:
                inplanes = channels_list[i]
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=channels_list[i+1]//block.expansion,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=self.with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                avg_down=self.avg_down,
                dcn=dcn,
                plugins=stage_plugins,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation,
                init_cfg=block_init_cfg)
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
    
    def forward(self, x):
        """Forward function."""
        curr_x = x[-1]
        for i, layer_name in enumerate(self.res_layers):
            if i != 0:
                curr_x = torch.cat([x[-(i+1)], curr_x], dim=1)
            res_layer = getattr(self, layer_name)
            curr_x = res_layer(curr_x)
        return curr_x

class ResUNet_MMSeg(nn.Module):
    def __init__(self, cfgs, out_dim=256):
        super(ResUNet_MMSeg, self).__init__()
        self.backbone = ResNet_encoder(
            depth=50, 
            in_channels=3,
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            out_indices=(0, 1), # useless here,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            deep_stem=True,
            avg_down=False,
        )
        if cfgs.decoder_head_1.block_type == 'Bottleneck_exp_rate_4':
            decoder_block = Bottleneck
        elif cfgs.decoder_head_1.block_type == 'Bottleneck_exp_rate_1':
            decoder_block = Bottleneck_exp_rate_1
        elif cfgs.decoder_head_1.block_type == 'Bottleneck_exp_rate_2':
            decoder_block = Bottleneck_exp_rate_2
        elif cfgs.decoder_head_1.block_type == 'BasicBlock_exp_rate_4':
            decoder_block = BasicBlock_exp_rate_4
        elif cfgs.decoder_head_1.block_type == 'BasicBlock_exp_rate_2':
            decoder_block = BasicBlock_exp_rate_2
        elif cfgs.decoder_head_1.block_type == 'BasicBlock_exp_rate_1':
            decoder_block = BasicBlock
        decoder_head_1_num_stage = len(cfgs.decoder_head_1.stage_blocks)
        decoder_head_1_strides = [1] * decoder_head_1_num_stage
        decoder_head_1_dilations = [1] * decoder_head_1_num_stage
        self.decoder_head_1 = ResNet_decoder(
            channels_list=cfgs.decoder_head_1.channels_list, # [2048, 1024, 512, 256, 64]
            stage_blocks=cfgs.decoder_head_1.stage_blocks, # [3, 6, 4 ,3]
            block=decoder_block,
            strides=decoder_head_1_strides,
            dilations=decoder_head_1_dilations,
            style='pytorch',
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            contract_dilation=True,
            upsample_cfg=cfgs.decoder_head_1.upsample_cfg,
            last_layer_num_blocks=cfgs.decoder_head_1.last_layer_num_blocks,
        )
        self.backbone_out_channels = [64, 256, 512, 1024, 2048]
        self.decoder_head_1_out_channels = [1024, 512, 256, 64, 64] # [14x14, 28x28, 56x56, 112x112, 112x112]
        self.c_fc = nn.Sequential(
                conv1x1(self.backbone_out_channels[-1], out_dim),
                nn.BatchNorm2d(out_dim),
            )
        self.f_layer = cfgs.f_layer
        self.f_fc = nn.Sequential(
                conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                nn.BatchNorm2d(out_dim),
            )
    
    def forward(self, x):
        middle_x = self.backbone(x)
        c_ebds_output = self.c_fc(middle_x[-1])
        out_x = self.decoder_head_1(middle_x)
        f_ebds_output = self.f_fc(out_x[self.f_layer - 1])
        return c_ebds_output, f_ebds_output

class ResUNet_MMSegv2(nn.Module):
    def __init__(self, cfgs, out_dim=256):
        super(ResUNet_MMSegv2, self).__init__()
        self.backbone = ResNet_encoder(
            depth=50, 
            in_channels=3,
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            out_indices=(0, 1), # useless here,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            deep_stem=True,
            avg_down=False,
        )
        if cfgs.decoder_head_1.block_type == 'Bottleneck_exp_rate_4':
            decoder_block = Bottleneck
        elif cfgs.decoder_head_1.block_type == 'Bottleneck_exp_rate_1':
            decoder_block = Bottleneck_exp_rate_1
        elif cfgs.decoder_head_1.block_type == 'Bottleneck_exp_rate_2':
            decoder_block = Bottleneck_exp_rate_2
        elif cfgs.decoder_head_1.block_type == 'BasicBlock_exp_rate_4':
            decoder_block = BasicBlock_exp_rate_4
        elif cfgs.decoder_head_1.block_type == 'BasicBlock_exp_rate_2':
            decoder_block = BasicBlock_exp_rate_2
        elif cfgs.decoder_head_1.block_type == 'BasicBlock_exp_rate_1':
            decoder_block = BasicBlock
        decoder_head_1_num_stage = len(cfgs.decoder_head_1.stage_blocks)
        decoder_head_1_strides = [1] * decoder_head_1_num_stage
        decoder_head_1_dilations = [1] * decoder_head_1_num_stage
        self.decoder_head_1 = ResNet_decoder(
            channels_list=cfgs.decoder_head_1.channels_list, # [2048, 1024, 512, 256, 64]
            stage_blocks=cfgs.decoder_head_1.stage_blocks, # [3, 6, 4 ,3]
            block=decoder_block,
            strides=decoder_head_1_strides,
            dilations=decoder_head_1_dilations,
            style='pytorch',
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            contract_dilation=True,
            upsample_cfg=cfgs.decoder_head_1.upsample_cfg,
            last_layer_num_blocks=cfgs.decoder_head_1.last_layer_num_blocks,
        )
        self.backbone_out_channels = [64, 256, 512, 1024, 2048]
        self.decoder_head_1_out_channels = [1024, 512, 256, 64, 64] # [14x14, 28x28, 56x56, 112x112, 112x112]
        self.layer1 = nn.Sequential(
            conv1x1(self.backbone_out_channels[4], out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.layer2 = nn.Sequential(
            conv1x1(self.decoder_head_1_out_channels[1], out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.layer3 = nn.Sequential(
            conv1x1(self.decoder_head_1_out_channels[2], out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.layer4 = nn.Sequential(
            conv1x1(self.decoder_head_1_out_channels[4], out_dim),
            nn.BatchNorm2d(out_dim),
        )
    
    def forward(self, x):
        middle_x = self.backbone(x)
        feats_1 = middle_x[-1].clone()
        out_x = self.decoder_head_1(middle_x)
        feats_2 = out_x[1]
        feats_3 = out_x[2]
        feats_4 = out_x[4]
        feats_1_out = self.layer1(feats_1)
        feats_2_out = self.layer2(feats_2)
        feats_3_out = self.layer3(feats_3)
        feats_4_out = self.layer4(feats_4)
        return feats_1_out, feats_2_out, feats_3_out, feats_4_out

class ResUNet_MMSegv3(nn.Module):
    def __init__(self, cfgs, out_dim=256):
        super(ResUNet_MMSegv3, self).__init__()
        self.backbone = ResNet_encoder(
            depth=50, 
            in_channels=3,
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            out_indices=(0, 1), # useless here,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            deep_stem=True,
            avg_down=False,
        )
        self.decoder_head_1 = ResNet_decoder(
            channels_list=[2048, 1024, 512, 256, 64],
            stage_blocks=[3, 6, 4, 3], # [3, 6, 4 ,3]
            block=Bottleneck_exp_rate_2,
            strides=[1, 1, 1, 1],
            dilations=[1, 1, 1, 1],
            style='pytorch',
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            contract_dilation=True,
            upsample_cfg=dict(type='InterpConv'),
            last_layer_num_blocks=2,
        )
        self.decoder_head_2 = ResNet_encoder_v2(
            channels_list=[64, 256, 512, 1024, 2048],
            stage_blocks=[3, 6, 4, 3],
            block=Bottleneck_exp_rate_2,
            strides=[2, 2, 2, 2],
            dilations=[1, 1, 1, 1],
            style='pytorch',
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            contract_dilation=True,
        )
        self.backbone_out_channels = [64, 256, 512, 1024, 2048]
        self.decoder_head_1_out_channels = [1024, 512, 256, 64, 64] # [14x14, 28x28, 56x56, 112x112, 112x112]
        self.layer1 = nn.Sequential(
            conv1x1(self.backbone_out_channels[4], out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.layer2 = nn.Sequential(
            conv1x1(self.decoder_head_1_out_channels[1], out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.layer3 = nn.Sequential(
            conv1x1(self.decoder_head_1_out_channels[2], out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.layer4 = nn.Sequential(
            conv1x1(self.decoder_head_1_out_channels[4], out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.layer5 = nn.Sequential(
            conv1x1(self.backbone_out_channels[4], out_dim),
            nn.BatchNorm2d(out_dim),
        )

        self.multi_layer_aggregator = False
        if 'multi_layer_aggregator' in cfgs.keys() and cfgs.multi_layer_aggregator:
            self.multi_layer_aggregator = True
            self.layer2_2 = nn.Sequential(
                conv1x1(self.decoder_head_1_out_channels[1], out_dim),
                nn.BatchNorm2d(out_dim),
            )
            self.layer3_2 = nn.Sequential(
                conv1x1(self.decoder_head_1_out_channels[2], out_dim),
                nn.BatchNorm2d(out_dim),
            )
            self.layer4_2 = nn.Sequential(
                conv1x1(self.decoder_head_1_out_channels[4], out_dim),
                nn.BatchNorm2d(out_dim),
            )
    
    def forward(self, x):
        middle_x = self.backbone(x)
        feats_1 = middle_x[-1].clone()
        out_x = self.decoder_head_1(middle_x)
        feats_2 = out_x[1]
        feats_3 = out_x[2]
        feats_4 = out_x[4]
        feats_1_out = self.layer1(feats_1)
        feats_2_out = self.layer2(feats_2)
        feats_3_out = self.layer3(feats_3)
        feats_4_out = self.layer4(feats_4)
        if self.multi_layer_aggregator:
            feats_2_2_out = self.layer2_2(feats_2)
            feats_3_2_out = self.layer3_2(feats_3)
            feats_4_2_out = self.layer4_2(feats_4)
        out_x = list(out_x)
        out_x.pop(3)
        feats_5 = self.decoder_head_2(out_x)
        feats_5_out = self.layer5(feats_5)
        if self.multi_layer_aggregator:
            return feats_1_out, feats_2_out, feats_2_2_out, feats_3_out, feats_3_2_out, feats_4_out, feats_4_2_out, feats_5_out
        else:
            return feats_1_out, feats_2_out, feats_3_out, feats_4_out, feats_5_out
    
class ResUNet_MMSegv4(nn.Module):
    def __init__(self, cfgs, out_dim=256):
        super(ResUNet_MMSegv4, self).__init__()
        self.backbone = ResNet_encoder(
            depth=50, 
            in_channels=3,
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            out_indices=(0, 1), # useless here,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            deep_stem=True,
            avg_down=False,
        )
        if cfgs.decoder_head_1.block_type == 'Bottleneck_exp_rate_4':
            decoder_block = Bottleneck
        elif cfgs.decoder_head_1.block_type == 'Bottleneck_exp_rate_1':
            decoder_block = Bottleneck_exp_rate_1
        elif cfgs.decoder_head_1.block_type == 'Bottleneck_exp_rate_2':
            decoder_block = Bottleneck_exp_rate_2
        elif cfgs.decoder_head_1.block_type == 'BasicBlock_exp_rate_4':
            decoder_block = BasicBlock_exp_rate_4
        elif cfgs.decoder_head_1.block_type == 'BasicBlock_exp_rate_2':
            decoder_block = BasicBlock_exp_rate_2
        elif cfgs.decoder_head_1.block_type == 'BasicBlock_exp_rate_1':
            decoder_block = BasicBlock
        decoder_head_1_num_stage = len(cfgs.decoder_head_1.stage_blocks)
        decoder_head_1_strides = [1] * decoder_head_1_num_stage
        decoder_head_1_dilations = [1] * decoder_head_1_num_stage
        self.decoder_head_1 = ResNet_decoder(
            channels_list=cfgs.decoder_head_1.channels_list, # [2048, 1024, 512, 256, 64]
            stage_blocks=cfgs.decoder_head_1.stage_blocks, # [3, 6, 4 ,3]
            block=decoder_block,
            strides=decoder_head_1_strides,
            dilations=decoder_head_1_dilations,
            style='pytorch',
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            contract_dilation=True,
            upsample_cfg=cfgs.decoder_head_1.upsample_cfg,
            last_layer_num_blocks=cfgs.decoder_head_1.last_layer_num_blocks,
        )
        self.backbone_out_channels = [64, 256, 512, 1024, 2048]
        self.decoder_head_1_out_channels = [1024, 512, 256, 64, 64] # [14x14, 28x28, 56x56, 112x112, 112x112]
        self.c_fc = nn.Sequential(
                conv1x1(self.backbone_out_channels[-1], out_dim),
                nn.BatchNorm2d(out_dim),
            )
        self.f_layer = cfgs.f_layer
        self.fc_type = 0
        if 'fc_type' in cfgs.keys():
            self.fc_type = cfgs.fc_type
            if cfgs.fc_type == 1:
                self.f_fc1 = nn.Sequential(
                    BasicBlock(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // BasicBlock.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                                type='Constant',
                                val=0,
                                override=dict(name='norm2')),
                    ),
                    BasicBlock(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // BasicBlock.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                                type='Constant',
                                val=0,
                                override=dict(name='norm2')
                                ),
                    ),
                    conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                    nn.BatchNorm2d(out_dim),
                )
                self.f_fc2 = nn.Sequential(
                    BasicBlock(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // BasicBlock.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                                type='Constant',
                                val=0,
                                override=dict(name='norm2')),
                    ),
                    BasicBlock(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // BasicBlock.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                                type='Constant',
                                val=0,
                                override=dict(name='norm2')
                                ),
                    ),
                    conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                    nn.BatchNorm2d(out_dim),
                )
            elif cfgs.fc_type == 2:
                self.f_fc1 = nn.Sequential(
                    BasicBlock(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // BasicBlock.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                                type='Constant',
                                val=0,
                                override=dict(name='norm2')),
                    ),
                    conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                    nn.BatchNorm2d(out_dim),
                )
                self.f_fc2 = nn.Sequential(
                    BasicBlock(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // BasicBlock.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                                type='Constant',
                                val=0,
                                override=dict(name='norm2')),
                    ),
                    conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                    nn.BatchNorm2d(out_dim),
                )
            elif cfgs.fc_type == 3:
                self.f_fc1 = nn.Sequential(
                    Bottleneck_exp_rate_2(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // Bottleneck_exp_rate_2.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3')),
                    ),
                    Bottleneck_exp_rate_2(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // Bottleneck_exp_rate_2.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3')),
                    ),
                    Bottleneck_exp_rate_2(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // Bottleneck_exp_rate_2.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3')),
                    ),
                    conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                    nn.BatchNorm2d(out_dim),
                )
                self.f_fc2 = nn.Sequential(
                    Bottleneck_exp_rate_2(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // Bottleneck_exp_rate_2.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3')),
                    ),
                    Bottleneck_exp_rate_2(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // Bottleneck_exp_rate_2.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3')),
                    ),
                    Bottleneck_exp_rate_2(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // Bottleneck_exp_rate_2.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3')),
                    ),
                    conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                    nn.BatchNorm2d(out_dim),
                )
            elif cfgs.fc_type == 4:
                self.f_fc1 = nn.Sequential(
                    BasicBlock(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // BasicBlock.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                                type='Constant',
                                val=0,
                                override=dict(name='norm2')),
                    ),
                    conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                    nn.BatchNorm2d(out_dim),
                )
                self.f_fc2 = nn.Sequential( # refering the Depth-Anything-v2's DPTHead
                    BasicBlock(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // BasicBlock.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                                type='Constant',
                                val=0,
                                override=dict(name='norm2')),
                    ),
                    nn.Conv2d(self.decoder_head_1_out_channels[self.f_layer - 1], 
                              self.decoder_head_1_out_channels[self.f_layer - 1] // 2, 
                              kernel_size=3,
                              stride=1,
                              padding=1), 
                )
                self.f_fc2_2 = nn.Sequential(            
                    nn.Conv2d(self.decoder_head_1_out_channels[self.f_layer - 1] // 2, 
                              self.decoder_head_1_out_channels[self.f_layer - 1], 
                              kernel_size=3, 
                              stride=1, 
                              padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(self.decoder_head_1_out_channels[self.f_layer - 1], 1, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(True),
                    nn.Identity(),
                    )
            elif cfgs.fc_type == 5:
                self.f_fc1 = nn.Sequential(
                    BasicBlock(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // BasicBlock.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                                type='Constant',
                                val=0,
                                override=dict(name='norm2')),
                    ),
                    conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                    nn.BatchNorm2d(out_dim),
                )
                self.f_fc2 = nn.Sequential( # refering the Depth-Anything-v2's DPTHead
                    BasicBlock(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // BasicBlock.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                                type='Constant',
                                val=0,
                                override=dict(name='norm2')),
                    ),
                    nn.Conv2d(self.decoder_head_1_out_channels[self.f_layer - 1], 
                              self.decoder_head_1_out_channels[self.f_layer - 1] // 2, 
                              kernel_size=3,
                              stride=1,
                              padding=1), 
                )
                self.f_fc2_2 = nn.Sequential(            
                    nn.Conv2d(self.decoder_head_1_out_channels[self.f_layer - 1] // 2, 
                              self.decoder_head_1_out_channels[self.f_layer - 1], 
                              kernel_size=3, 
                              stride=1, 
                              padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(self.decoder_head_1_out_channels[self.f_layer - 1], 1, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(True),
                    nn.Identity(),
                    )
                self.f_fc3 = nn.Sequential(
                    BasicBlock(
                        inplanes=self.decoder_head_1_out_channels[self.f_layer - 1],
                        planes=self.decoder_head_1_out_channels[self.f_layer - 1] // BasicBlock.expansion,
                        stride=1,
                        dilation=1,
                        downsample=None,
                        conv_cfg=None,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        style='pytorch',
                        with_cp=False,
                        dcn=None,
                        plugins=None,
                        init_cfg=dict(
                                type='Constant',
                                val=0,
                                override=dict(name='norm2')),
                    ),
                    conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                    nn.BatchNorm2d(out_dim),
                )
        else:
            self.f_fc1 = nn.Sequential(
                    conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                    nn.BatchNorm2d(out_dim),
                )
            self.f_fc2 = nn.Sequential(
                    conv1x1(self.decoder_head_1_out_channels[self.f_layer - 1], out_dim),
                    nn.BatchNorm2d(out_dim),
                )
    
    def forward(self, x):
        H, W = x.shape[2:]
        middle_x = self.backbone(x)
        c_ebds_output = self.c_fc(middle_x[-1])
        out_x = self.decoder_head_1(middle_x)
        f1_ebds_output = self.f_fc1(out_x[self.f_layer - 1])
        f2_ebds_output = self.f_fc2(out_x[self.f_layer - 1])
        if self.fc_type == 4:
            f2_ebds_output = F.interpolate(f2_ebds_output, size=(H, W), mode='bilinear', align_corners=True)
            f2_ebds_output = self.f_fc2_2(f2_ebds_output)
        elif self.fc_type == 5:
            f2_ebds_output = F.interpolate(f2_ebds_output, size=(H, W), mode='bilinear', align_corners=True)
            f2_ebds_output = self.f_fc2_2(f2_ebds_output)
            c_ebds_output = self.f_fc3(out_x[self.f_layer - 1])
        return c_ebds_output, f1_ebds_output, f2_ebds_output