import torch.nn as nn
from .resnetfpn_dp import ResNetFPN_16_4, ResNetFPN_8_2, ResNetFPN_MMSeg, ResNetFPN_MMSegv2
from .resnet_dp import ResNet18
from .cct_dp import CCT, CCTv2
from .resnetfpnv2_dp import ResNetFPNv2
from .aggregate_dp import aggregator
from .unet_dp import UNet_MMSeg
from .resunet_dp import ResUNet_MMSeg, ResUNet_MMSegv2, ResUNet_MMSegv4, ResUNet_MMSegv3
from .cmvpr_dp import generate_img_index_and_knn_and_coords_v3


class ImageEncoder(nn.Module):

    def __init__(self, backbone_type, backbone_cfgs, out_dim, out_layer):
        super(ImageEncoder, self).__init__()
        if backbone_type == 'ResFPN164':
            self.module = ResNetFPN_16_4(backbone_cfgs, out_dim)
        elif backbone_type == 'ResFPN82':
            self.module = ResNetFPN_8_2(backbone_cfgs, out_dim)
        elif backbone_type == 'Res18':
            self.module = ResNet18(backbone_cfgs, out_dim)
        elif backbone_type == 'CCT':
            self.module = CCT(backbone_cfgs, out_dim)
        elif backbone_type == 'ResFPNv2':
            self.module = ResNetFPNv2(backbone_cfgs, out_dim)
        elif backbone_type == 'CCTv2':
            self.module = CCTv2(backbone_cfgs, out_dim)
        elif backbone_type == 'ResFPNmmseg':
            self.module = ResNetFPN_MMSeg(backbone_cfgs, out_dim)
        elif backbone_type == 'ResFPNmmsegv2':
            self.module = ResNetFPN_MMSegv2(backbone_cfgs, out_dim)
        elif backbone_type == 'UNetmmseg':
            self.module = UNet_MMSeg(backbone_cfgs, out_dim)
        elif backbone_type == 'ResUNetmmseg':
            self.module = ResUNet_MMSeg(backbone_cfgs, out_dim)
        elif backbone_type == 'ResUNetmmsegv2':
            self.module = ResUNet_MMSegv2(backbone_cfgs, out_dim)
        elif backbone_type == 'ResUNetmmsegv3':
            self.module = ResUNet_MMSegv3(backbone_cfgs, out_dim)
        elif backbone_type == 'ResUNetmmsegv4':
            self.module = ResUNet_MMSegv4(backbone_cfgs, out_dim)
        else:
            raise ValueError('Image Encoder Backbone type not supported')
        self.layer = out_layer
    
    def forward(self, x):
        if self.layer == 1:
            feats, _ = self.module(x)
            return feats
        elif self.layer == 2:
            _, feats = self.module(x)
            return feats
        elif self.layer == 3:
            c_feats, f_feats = self.module(x)
            return c_feats, f_feats
        elif self.layer == 4:
            feats, _, _ = self.module(x)
            return feats
        elif self.layer == 5:
            _, feats, _ = self.module(x)
            return feats
        elif self.layer == 6:
            _, _, feats = self.module(x)
            return feats
        elif self.layer == 7:
            feats_1, feats_2, feats_3, feats_4 = self.module(x)
            return feats_1, feats_2, feats_3, feats_4
        elif self.layer == 8:
            c_feats, f1_feats, f2_feats = self.module(x)
            return c_feats, f1_feats, f2_feats
        elif self.layer == 9:
            feats_1, feats_2, feats_3, feats_4, feats_5 = self.module(x)
            return feats_1, feats_2, feats_3, feats_4, feats_5
        elif self.layer == 10:
            feats_1, feats_2, feats_2_2, feats_3, feats_3_2, feats_4, feats_4_2, feats_5 = self.module(x)
            return feats_1, feats_2, feats_2_2, feats_3, feats_3_2, feats_4, feats_4_2, feats_5
        elif self.layer == 11:
            feats_1, feats_2, feats_3, feats_4 = self.module(x)
            return feats_4
        elif self.layer == 12:
            feats_1, feats_2, feats_3, feats_4 = self.module(x)
            return feats_1
        else:
            raise ValueError('Layer not supported')
        


class ImageNet(nn.Module):

    def __init__(self, config, out_dim):
        super(ImageNet, self).__init__()
        self.backbone = ImageEncoder(config.backbone_type, config.backbone_config, out_dim, config.out_layer)
        self.aggregator = aggregator(config.aggregate_type, config.aggregate_config, out_dim)
        self.aggregate_type = config.aggregate_type
        if self.aggregate_type == 'PoS_GeM':
            self.resolution_list = config.resolution_list
    
    
    def forward(self, x):
        data_output = {}
        feats = self.backbone(x['images'])
        if self.aggregate_type == 'PoS_GeM':
            index_list, knn_index_list, img_mesh_list = generate_img_index_and_knn_and_coords_v3(feats.shape[0], self.resolution_list, feats.device)
            feats = feats.flatten(2)
            data_output['embeddings'] = self.aggregator(feats, index_list=index_list, coords_list=img_mesh_list)
        else:
            data_output['embeddings'] = self.aggregator(feats)
        return data_output

class RenderNet(nn.Module):

    def __init__(self, config, out_dim):
        super(RenderNet, self).__init__()
        self.backbone = ImageEncoder(config.backbone_type, config.backbone_config, out_dim, config.out_layer)
        self.aggregator = aggregator(config.aggregate_type, config.aggregate_config, out_dim)
    
    def forward(self, x):
        data_output = {}
        feats = self.backbone(x['render_imgs'])
        data_output['embeddings'] = self.aggregator(feats)
        return data_output