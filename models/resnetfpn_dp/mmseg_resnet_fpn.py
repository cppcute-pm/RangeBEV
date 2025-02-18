from mmseg.models.necks import FPN
from mmseg.models.decode_heads import FPNHead
from mmseg.models.utils import resize
from mmengine.model import BaseModule
from mmseg.models.backbones import ResNetV1c
from torch import nn



class FPN_m1(FPN):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        BaseModule.__init__(init_cfg)
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + resize(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + resize(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

class FPNHead_v2(FPNHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """


    # def __init__(self,
    #              c_layer,
    #              f_layer,
    #              in_channels, 
    #              in_index,
    #              feature_strides, 
    #              channels,
    #              dropout_ratio,
    #              num_classes,
    #              norm_cfg,
    #              align_corners,
    #              loss_decode,
    #              **kwargs):
    #     super(self, FPNHead_v2).__init__(feature_strides=feature_strides,
    #                                      in_channels=in_channels,
    #                                      in_index=in_index,
    #                                      channels=channels,
    #                                      dropout_ratio=dropout_ratio,
    #                                      num_classes=num_classes,
    #                                      norm_cfg=norm_cfg,
    #                                      align_corners=align_corners,
    #                                      loss_decode=loss_decode,
    #                                      **kwargs,
    #                                      )

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        output_list = [output]
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output_list.append(output)
        return output_list

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class ResNetFPN_MMSeg(nn.Module):

    def __init__(self, cfgs, out_dim=256):
        super(ResNetFPN_MMSeg, self).__init__()
        self.backbone = ResNetV1c(depth=cfgs.backbone.depth, # 50
                                  in_channels=cfgs.backbone.in_channels, # 3
                                  num_stages=cfgs.backbone.num_stages, # 4
                                  strides=cfgs.backbone.strides, # (1, 2, 2, 2)
                                  dilations=cfgs.backbone.dilations, # (1, 1, 1, 1)
                                  out_indices=cfgs.backbone.out_indices, # (0, 1, 2, 3),
                                  norm_cfg=cfgs.norm_cfg, # dict(type='SyncBN', requires_grad=True)
                                  norm_eval=cfgs.backbone.norm_eval, # False
                                  style=cfgs.backbone.style, # 'pytorch'
                                  contract_dilation=cfgs.backbone.contract_dilation, # True
                                  )
        self.neck = FPN(in_channels=cfgs.neck.in_channels, # [256 (56x56), 512 (28x28), 1024 (14x14), 2048 (7x7)]
                        out_channels=cfgs.neck.out_channels, # 256
                        num_outs=cfgs.neck.num_outs, # 4
                        ) 
        self.decode_head = FPNHead_v2(in_channels=cfgs.decode_head.in_channels, # [256, 256, 256, 256]
                                   in_index=cfgs.decode_head.in_index, # [0, 1, 2, 3]
                                   feature_strides=cfgs.decode_head.feature_strides, # [4, 8, 16, 32]
                                   channels=cfgs.decode_head.channels, # 128
                                   dropout_ratio=cfgs.decode_head.dropout_ratio, # 0.1
                                    num_classes=2,
                                    norm_cfg=cfgs.norm_cfg, # dict(type='SyncBN', requires_grad=True)
                                    align_corners=cfgs.decode_head.align_corners, # False
                                    loss_decode=cfgs.decode_head.loss_decode, # dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                                    )
        self.f_layer = cfgs.f_layer
        self.c_layer = cfgs.c_layer
        self.num_stages = cfgs.backbone.num_stages
        if 'new_definition' in cfgs.keys() and cfgs.new_definition:
            self.new_definition = True
            self.all_channels = cfgs.neck.in_channels + [cfgs.decode_head.channels] + [cfgs.decode_head.channels] + [cfgs.decode_head.channels] + [cfgs.decode_head.channels]
            self.c_fc = nn.Sequential(
                conv1x1(self.all_channels[self.c_layer - 1], out_dim),
                nn.BatchNorm2d(out_dim),
            )
            self.f_fc = nn.Sequential(
                conv1x1(self.all_channels[self.f_layer - 1], out_dim),
                nn.BatchNorm2d(out_dim),
            )
        else:
            self.new_definition = False
            self.c_fc = nn.Sequential(
                conv1x1(cfgs.decode_head.channels, out_dim),
                nn.BatchNorm2d(out_dim),
            )
            self.f_fc = nn.Sequential(
                conv1x1(cfgs.decode_head.channels, out_dim),
                nn.BatchNorm2d(out_dim),
            )

    
    def forward(self, x):
        if not self.new_definition:
            x = self.backbone(x)
            x = self.neck(x)
            feature_list = self.decode_head(x)
            c_ebds_output = self.c_fc(feature_list[self.c_layer - 1])
            f_ebds_output = self.f_fc(feature_list[self.f_layer - 1])
            return c_ebds_output, f_ebds_output
        else:
            c_ebds_output = None
            f_ebds_output = None
            x = self.backbone(x)
            if self.c_layer <= self.num_stages:
                c_ebds_output = self.c_fc(x[self.c_layer - 1])
            if self.f_layer <= self.num_stages:
                f_ebds_output = self.f_fc(x[self.f_layer - 1])
            if c_ebds_output is not None and f_ebds_output is not None:
                return c_ebds_output, f_ebds_output
            x = self.neck(x)
            feature_list = self.decode_head(x)
            if self.c_layer > self.num_stages:
                c_ebds_output = self.c_fc(feature_list[self.c_layer - self.num_stages - 1])
            if self.f_layer > self.num_stages:
                f_ebds_output = self.f_fc(feature_list[self.f_layer - self.num_stages - 1])
            return c_ebds_output, f_ebds_output

class ResNetFPN_MMSegv2(nn.Module):

    def __init__(self, cfgs, out_dim=256):
        super(ResNetFPN_MMSegv2, self).__init__()
        self.backbone = ResNetV1c(depth=cfgs.backbone.depth, # 50
                                  in_channels=cfgs.backbone.in_channels, # 3
                                  num_stages=cfgs.backbone.num_stages, # 4
                                  strides=cfgs.backbone.strides, # (1, 2, 2, 2)
                                  dilations=cfgs.backbone.dilations, # (1, 1, 1, 1)
                                  out_indices=cfgs.backbone.out_indices, # (0, 1, 2, 3),
                                  norm_cfg=cfgs.norm_cfg, # dict(type='SyncBN', requires_grad=True)
                                  norm_eval=cfgs.backbone.norm_eval, # False
                                  style=cfgs.backbone.style, # 'pytorch'
                                  contract_dilation=cfgs.backbone.contract_dilation, # True
                                  )
        self.neck = FPN(in_channels=cfgs.neck.in_channels, # [256, 512, 1024, 2048]
                        out_channels=cfgs.neck.out_channels, # 256
                        num_outs=cfgs.neck.num_outs, # 4
                        ) 
        self.decode_head = FPNHead_v2(in_channels=cfgs.decode_head.in_channels, # [256, 256, 256, 256]
                                   in_index=cfgs.decode_head.in_index, # [0, 1, 2, 3]
                                   feature_strides=cfgs.decode_head.feature_strides, # [4, 8, 16, 32]
                                   channels=cfgs.decode_head.channels, # 128
                                   dropout_ratio=cfgs.decode_head.dropout_ratio, # 0.1
                                    num_classes=2,
                                    norm_cfg=cfgs.norm_cfg, # dict(type='SyncBN', requires_grad=True)
                                    align_corners=cfgs.decode_head.align_corners, # False
                                    loss_decode=cfgs.decode_head.loss_decode, # dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                                    )
        self.all_channels = cfgs.neck.in_channels + [cfgs.decode_head.channels] + [cfgs.decode_head.channels] + [cfgs.decode_head.channels] + [cfgs.decode_head.channels]
        self.layer1 = nn.Sequential(
            conv1x1(self.all_channels[3], out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.layer2 = nn.Sequential(
            conv1x1(self.all_channels[5], out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.layer3 = nn.Sequential(
            conv1x1(self.all_channels[6], out_dim),
            nn.BatchNorm2d(out_dim),
        )
        self.layer4 = nn.Sequential(
            conv1x1(self.all_channels[7], out_dim),
            nn.BatchNorm2d(out_dim),
        )

    
    def forward(self, x):
        x = self.backbone(x)
        feats_1 = x[-1].clone()
        x = self.neck(x)
        feature_list = self.decode_head(x)
        feats_2 = feature_list[1]
        feats_3 = feature_list[2]
        feats_4 = feature_list[3]
        feats_1_out = self.layer1(feats_1)
        feats_2_out = self.layer2(feats_2)
        feats_3_out = self.layer3(feats_3)
        feats_4_out = self.layer4(feats_4)
        return feats_1_out, feats_2_out, feats_3_out, feats_4_out


class ResNetFPN_MMSegv3(nn.Module):

    def __init__(self, cfgs, out_dim=256):
        super(ResNetFPN_MMSegv3, self).__init__()
        self.backbone = ResNetV1c(depth=cfgs.backbone.depth, # 50
                                  in_channels=cfgs.backbone.in_channels, # 3
                                  num_stages=cfgs.backbone.num_stages, # 4
                                  strides=cfgs.backbone.strides, # (1, 2, 2, 2)
                                  dilations=cfgs.backbone.dilations, # (1, 1, 1, 1)
                                  out_indices=cfgs.backbone.out_indices, # (0, 1, 2, 3),
                                  norm_cfg=cfgs.norm_cfg, # dict(type='SyncBN', requires_grad=True)
                                  norm_eval=cfgs.backbone.norm_eval, # False
                                  style=cfgs.backbone.style, # 'pytorch'
                                  contract_dilation=cfgs.backbone.contract_dilation, # True
                                  )
        self.neck = FPN(in_channels=cfgs.neck.in_channels, # [256, 512, 1024, 2048]
                        out_channels=cfgs.neck.out_channels, # 256
                        num_outs=cfgs.neck.num_outs, # 4
                        ) 
        self.decode_head = FPNHead_v2(in_channels=cfgs.decode_head.in_channels, # [256, 256, 256, 256]
                                   in_index=cfgs.decode_head.in_index, # [0, 1, 2, 3]
                                   feature_strides=cfgs.decode_head.feature_strides, # [4, 8, 16, 32]
                                   channels=cfgs.decode_head.channels, # 128
                                   dropout_ratio=cfgs.decode_head.dropout_ratio, # 0.1
                                    num_classes=2,
                                    norm_cfg=cfgs.norm_cfg, # dict(type='SyncBN', requires_grad=True)
                                    align_corners=cfgs.decode_head.align_corners, # False
                                    loss_decode=cfgs.decode_head.loss_decode, # dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                                    )
        self.neck_v2 = FPN_m1(in_channels=cfgs.neck_v2.in_channels, # [256, 256, 256, 256]
                        out_channels=cfgs.neck_v2.out_channels, # 256
                        num_outs=cfgs.neck_v2.num_outs, # 4
                        )
        self.backbone_v2 = ResNet_m1()
