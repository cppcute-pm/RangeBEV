import torch.nn as nn
from .fpt_dp import fast_point_transformer
from .pointmlp_dp import PointMLP
from .dgcnn_dp import DGCNN
from .pointnext_dp import PointNext, PointNextv2, PointNextv3, PointNextv4
from .aggregate_dp import aggregator
from .SLidR_dp import MinkUNet, VoxelNet
from .cmvpr_dp import (generate_pc_index_and_coords_v1, 
                       generate_pc_index_and_coords_v2)
import torch


class PcEncoder(nn.Module):

    def __init__(self, backbone_type, backbone_cfgs, out_dim, out_layer):
        super(PcEncoder, self).__init__()
        if backbone_type == 'FPT':
            self.module = fast_point_transformer(backbone_cfgs, out_dim)
        elif backbone_type == 'PointMLP':
            self.module = PointMLP(backbone_cfgs, out_dim)
        elif backbone_type == 'DGCNN':
            self.module = DGCNN(backbone_cfgs, out_dim)
        elif backbone_type == 'PointNext':
            self.module = PointNext(backbone_cfgs, out_dim)
        elif backbone_type == 'MinkUNet':
            self.module = MinkUNet(backbone_cfgs, out_dim)
        elif backbone_type == 'PointNextv2':
            self.module = PointNextv2(backbone_cfgs, out_dim)
        elif backbone_type == 'PointNextv3':
            self.module = PointNextv3(backbone_cfgs, out_dim)
        elif backbone_type == 'PointNextv4':
            self.module = PointNextv4(backbone_cfgs, out_dim)
        else:
            raise ValueError('pc Encoder Backbone type not supported')
        self.layer = out_layer
        self.module_type = backbone_type
        self.backbone_cfgs = backbone_cfgs
        self.pointnextv3_op = backbone_cfgs.pointnextv3_op
    
    def forward(self, x):
        if self.module_type == 'FPT' or self.module_type == 'MinkUNet':
            f_feats, c_feats, f_points, c_points, f_masks, c_masks = self.module.forward_plus(x)
        elif self.module_type == 'PointNextv2':
            feats_1, feats_2, feats_3, feats_4, points_1, points_2, points_3, points_4 = self.module(x)
        elif self.module_type == 'PointNextv3':
            if self.pointnextv3_op == 1:
                f1_feats, f2_feats, c_feats, f_points, c_points, f_masks, c_masks = self.module(x)
            elif self.pointnextv3_op == 2:
                f1_feats, f2_feats, c_feats, points_1, points_2, points_3, points_4 = self.module(x)
            else:
                raise ValueError('PointNextv3 operation not supported')
        elif self.module_type == 'PointNextv4':
            if 'multi_layer_aggregator' in self.backbone_cfgs.keys() and self.backbone_cfgs.multi_layer_aggregator:
                (feats_1, 
                 feats_2, 
                 feats_2_2, 
                 feats_3, 
                 feats_3_2, 
                 feats_4, 
                 feats_4_2, 
                 feats_5, 
                 points_1, 
                 points_2, 
                 points_3, 
                 points_4, 
                 points_5) = self.module(x)
            else:
                feats_1, feats_2, feats_3, feats_4, feats_5, points_1, points_2, points_3, points_4, points_5 = self.module(x)
        else:
            f_feats, c_feats, f_points, c_points, f_masks, c_masks = self.module(x)
        if self.layer == 1:
            feats = c_feats
            masks = c_masks
            return feats, masks
        elif self.layer == 2:
            feats = f_feats
            masks = f_masks
            return feats, masks
        elif self.layer == 3:
            return f_feats, c_feats, f_points, c_points, f_masks, c_masks
        elif self.layer == 4:
            return feats_4, points_1, points_2, points_3, points_4
        elif self.layer == 5:
            return f_feats, f_points
        elif self.layer == 6:
            return f1_feats, f2_feats, c_feats, points_1, points_2, points_3, points_4
        elif self.layer == 7:
            return feats_1, feats_2, feats_3, feats_4, points_1, points_2, points_3, points_4
        elif self.layer == 8:
            return f1_feats, f2_feats, c_feats, f_points, c_points
        elif self.layer == 9:
            return feats_1, feats_2, feats_3, feats_4, feats_5, points_1, points_2, points_3, points_4, points_5
        elif self.layer == 10:
            return feats_1, feats_2, feats_2_2, feats_3, feats_3_2, feats_4, feats_4_2, feats_5, points_1, points_2, points_3, points_4, points_5
        elif self.layer == 11:
            masks = torch.ones((feats_4.shape[0], feats_4.shape[2]), device=feats_4.device, dtype=torch.bool)
            return feats_4, masks
        elif self.layer == 12:
            return feats_1, points_1
        else:
            raise ValueError('PCNet Layer type not supported')
        


class PCNet(nn.Module):

    def __init__(self, config, out_dim):
        super(PCNet, self).__init__()
        self.backbone = PcEncoder(config.backbone_type, config.backbone_config, out_dim, config.out_layer)
        self.aggregator = aggregator(config.aggregate_type, config.aggregate_config, out_dim)
        self.aggregate_type = config.aggregate_type
        if self.aggregate_type == 'PoS_GeM':
            self.coords_type = config.coords_type
            if self.coords_type == 'num_list':
                self.coords_num_list = config.coords_num_list
    
    def forward(self, x):
        data_output = {}
        if self.aggregate_type == 'PoS_GeM':
            if self.coords_type == 'backbone':
                feats, coords_4, coords_3, coords_2, coords_1 = self.backbone(x['clouds'])
                coords_list = [coords_1, coords_2, coords_3, coords_4]
                index_list, coords_list = generate_pc_index_and_coords_v2(coords_list)
            elif self.coords_type == 'num_list':
                feats, coords = self.backbone(x['clouds'])
                index_list, coords_list = generate_pc_index_and_coords_v1(self.coords_num_list, coords)
            elif self.coords_type == 'backbone_v2':
                feats, coords = self.backbone(x['clouds'])
                coords_list = [coords]
                index_list, coords_list = generate_pc_index_and_coords_v2(coords_list)
            else:
                raise ValueError('coords_type not supported')
            data_output['embeddings'] = self.aggregator(feats, mask=None, index_list=index_list, coords_list=coords_list)
        else:
            feats, masks = self.backbone(x['clouds'])
            data_output['embeddings'] = self.aggregator(feats, masks)
        return data_output