import torch
import torch.nn as nn
from copy import deepcopy
import math

from .swin_transformer_dp import WindowMSA, WindowMCA, SwinBlock, CwindowBlock
from .point_transformer_v2_dp import GroupedVectorSA, GroupedVectorCA, CABlockSeq, SABlockSeq
from .residual_attention import ResidualSelfAttentionBlock_v2
from .scatter_gather import (generate_img_index_and_knn_and_coords_v1,
                             generate_img_index_and_knn_and_coords_v2,
                             generate_img_index_and_knn_and_coords_v3,
                             inverse_mapping,
                             generate_pc_index_and_coords_v2,
                             generate_pc_index_and_coords_v3)
import torch_scatter
from pykeops.torch import LazyTensor
import time


# class ImgAttnRe(nn.Module):

#     def __init__(self, cfgs, out_dim):
#         super(ImgAttnRe, self).__init__()
#         assert len(cfgs.high_score_ratio_list) == len(cfgs.low_score_ratio_list)
#         assert len(cfgs.ca_cfgs_list) == len(cfgs.sa_cfgs_list)
#         self.high_score_ratio_list = cfgs.high_score_ratio_list
#         self.num_neighbor_list = cfgs.num_neighbor_list
#         self.low_score_ratio_list = cfgs.low_score_ratio_list
#         self.out_dim = out_dim
#         self.depth_in_one_layer = cfgs.depth_in_one_layer
#         self.first_SA_blocks = nn.ModuleList()
#         self.SA_blocks_1 = nn.ModuleList()
#         self.SA_blocks_2 = nn.ModuleList()
#         self.CA_blocks = nn.ModuleList()
#         self.cls_emb = nn.Parameter(torch.zeros(1, 1, out_dim),
#                                        requires_grad=True)
#         self.pos_emb = nn.Parameter(self.sinusoidal_embedding(cfgs.sequence_length + 1, out_dim),
#                                     requires_grad=False)
#         first_SA_block = ResidualSelfAttentionBlock_v2(out_dim,
#                                                        cfgs.first_SA_block_cfgs)
#         self.layer_num = len(cfgs.sa_cfgs_list)
#         for i in range(cfgs.first_SA_block_cfgs.num_blocks):
#             self.first_SA_blocks.append(deepcopy(first_SA_block))
#         for i in range(len(cfgs.sa_cfgs_list)):
#             curr_resolution_SA_blocks_1 = nn.ModuleList()
#             curr_resolution_SA_blocks_2 = nn.ModuleList()
#             for j in range(self.depth_in_one_layer):
#                 curr_depth_SA_blocks_1 = nn.ModuleList()
#                 curr_depth_SA_blocks_2 = nn.ModuleList()
#                 curr_depth_SA_blocks_1.append(SwinBlock(out_dim, cfgs.sa_cfgs_list[i][0], shift=False))
#                 curr_depth_SA_blocks_1.append(SwinBlock(out_dim, cfgs.sa_cfgs_list[i][0], shift=True))
#                 curr_depth_SA_blocks_2.append(SwinBlock(out_dim, cfgs.sa_cfgs_list[i][1], shift=False))
#                 curr_depth_SA_blocks_2.append(SwinBlock(out_dim, cfgs.sa_cfgs_list[i][1], shift=True))
#                 curr_resolution_SA_blocks_1.append(curr_depth_SA_blocks_1)
#                 curr_resolution_SA_blocks_2.append(curr_depth_SA_blocks_2)
#             self.SA_blocks_1.append(curr_resolution_SA_blocks_1)
#             self.SA_blocks_2.append(curr_resolution_SA_blocks_2)
#         for i in range(len(cfgs.ca_cfgs_list)):
#             curr_resolution_CA_blocks = nn.ModuleList()
#             for j in range(self.depth_in_one_layer):
#                 curr_resolution_CA_blocks.append(CwindowBlock(out_dim, cfgs.ca_cfgs_list[i]))
#             self.CA_blocks.append(curr_resolution_CA_blocks)
#         self.attn_type = cfgs.attn_type
#         self.attn_use_type = cfgs.attn_use_type
#         self.feat_attn_type = cfgs.feat_attn_type


#     def forward(self, img_feats_list):
#         B = img_feats_list[0].size(0)
#         device = img_feats_list[0].device
#         aggr_feat_list = []
#         to_aggr_feat_idx_list = []
#         coarsest_feat = img_feats_list[0].flatten(2).permute(2, 0, 1) # [B, C, H, W] -> [B, C, H*W] -> [H*W, B, C]
#         assert coarsest_feat.size(0) == (self.pos_emb.size(1) - 1)
#         resolution_list = [temp_feats.shape[2:] for temp_feats in img_feats_list]
#         resolution_list.insert(0, (1, 1))
#         resolution_list = resolution_list[::-1]
#         index_list, knn_index_list_1, knn_index_list_2, img_mesh_list = generate_img_index_and_knn_and_coords_v2(B, 
#                                                                                              resolution_list, 
#                                                                                              device, 
#                                                                                              self.num_neighbor_list)

#         cls_token = self.cls_emb.expand(-1, coarsest_feat.size(1), -1) # [1, B, C]
#         coarsest_feat = torch.cat([cls_token, coarsest_feat], dim=0) + self.pos_emb.permute(1, 0, 2) # [H*W+1, B, C]
#         for i in range(len(self.first_SA_blocks)):
#             coarsest_feat, attn_weights = self.first_SA_blocks[i](coarsest_feat) # [H*W + 1, B, C]、[B, H*W + 1, H*W + 1]
        

#         aggr_feat_list.append(coarsest_feat[0, :, :])
#         coarsest_feat = coarsest_feat[1:, :, :] # [H*W, B, C]
#         attn_inuse = attn_weights[:, 0, 1:] # [B, H*W]
#         attn_inuse_sorted_idxs = torch.argsort(attn_inuse, dim=1, descending=True) # [B, H*W]
#         to_aggr_feat_idx = attn_inuse_sorted_idxs[:, :int(attn_inuse.size(1) * self.high_score_ratio_list[0])]
#         to_aggr_feat_idx_list.append(to_aggr_feat_idx)

#         # the second dim of attn_inuse and attn_inuse_sorted_idxs may not be the same
#         for i in range(self.layer_num):
#             to_re_feat_idx = attn_inuse_sorted_idxs[:, int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[i]): int(attn_inuse_sorted_idxs.size(1) * self.low_score_ratio_list[i])] # [B, num_re]
#             curr_attn_threshold = []
#             curr_attn_threshold.append(
#                 torch.gather(
#                     attn_inuse,
#                     dim=1,
#                     index=attn_inuse_sorted_idxs[:, int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[i]):int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[i])+1]))
#             curr_attn_threshold.append(
#                 torch.gather(
#                     attn_inuse,
#                     dim=1,
#                     index=attn_inuse_sorted_idxs[:, int(attn_inuse_sorted_idxs.size(1) * self.low_score_ratio_list[i]):int(attn_inuse_sorted_idxs.size(1) * self.low_score_ratio_list[i])+1]))
#             curr_attn_threshold = torch.cat(curr_attn_threshold, dim=1) # [B, 2]
#             to_all_feat_mesh = img_mesh_list[-(i+2)]
#             to_re_feat_mesh = torch.gather(to_all_feat_mesh,
#                                         dim=1,
#                                         index=to_re_feat_idx.unsqueeze(-1).expand(-1, -1, 2)) # [B, num_re, 2]
#             re_to_all_feat_mesh_dist = torch.cdist(to_re_feat_mesh, to_all_feat_mesh, p=2.0) # [B, num_re, H*W]
#             to_re_feat_idx_inuse = torch.topk(re_to_all_feat_mesh_dist, 
#                                                 k=self.num_neighbor_list[i], 
#                                                 dim=-1, 
#                                                 largest=False, 
#                                                 sorted=False)[1] # [B, num_re, curr_num_neighbor]
#             to_re_feat_idx_inuse = torch.sort(to_re_feat_idx_inuse, dim=-1, descending=False)[0] # ensure the order of the index same as the space distribution
#             num_re, curr_num_neighbor = to_re_feat_idx_inuse.size(1), to_re_feat_idx_inuse.size(2)
#             to_re_feat_attn_inuse = torch.gather(attn_inuse,
#                                                 dim=1,
#                                                 index=to_re_feat_idx_inuse.flatten(1)) # [B, num_re * curr_num_neighbor]
#             assert torch.count_nonzero(to_re_feat_attn_inuse == 0.) == 0
#             to_re_feat_attn_inuse = to_re_feat_attn_inuse.reshape(B, -1, curr_num_neighbor) # [B, num_re, curr_num_neighbor]
#             to_re_feat_inuse = torch.gather(coarsest_feat,
#                                             dim=0,
#                                             index=to_re_feat_idx_inuse.permute(1, 2, 0).flatten(0, 1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [num_re * curr_num_neighbor, B, C]
#             to_re_feat_inuse = to_re_feat_inuse.permute(1, 0, 2).reshape(B, num_re, curr_num_neighbor, self.out_dim).reshape(-1, curr_num_neighbor, self.out_dim) # [B * num_re, curr_num_neighbor, C]
#             num_k_1 = knn_index_list_1[-(i+2)].shape[1]
#             next_to_re_feat_idx_1 = torch.gather(knn_index_list_1[-(i+2)],
#                                                 dim=-1,
#                                                 index=to_re_feat_idx.unsqueeze(1).expand(-1, num_k_1, -1)) # [B, num_k_1, num_re]


#             num_k_2 = knn_index_list_2[-(i+2)].shape[1]
#             next_to_re_feat_idx = torch.gather(knn_index_list_2[-(i+2)], # [B, num_k_2, H*W]
#                                                 dim=-1,
#                                                 index=to_re_feat_idx.unsqueeze(1).expand(-1, num_k_2, -1)) # [B, num_k_2, num_re]
#             next_H, next_W = img_feats_list[i+1].shape[2:]
#             next_to_all_feat = img_feats_list[i+1].flatten(2).permute(0, 2, 1) # [B, C, next_H, next_W] -> [B, C, next_H*next_W] -> [B, next_H*next_W, C]

#             if self.feat_attn_type == 'all_feat':
#                 next_to_all_feat_num = torch.zeros_like(next_to_all_feat) # [B, next_H*next_W, C]
#                 next_to_re_feat_num = torch.ones((B, num_k_2, num_re, self.out_dim), dtype=next_to_all_feat_num.dtype, device=device) # [B, num_k_2, num_re, C]
#                 next_to_all_feat_num.scatter_(dim=1, 
#                                             index=next_to_re_feat_idx.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim), 
#                                             src=next_to_re_feat_num.flatten(1, 2),
#                                             reduce='add')
#                 next_to_all_feat_num = torch.where(next_to_all_feat_num == 0.0, torch.ones_like(next_to_all_feat_num), next_to_all_feat_num)
#                 for j in range(self.depth_in_one_layer):
#                     next_to_all_feat = self.SA_blocks_1[i][j][0](next_to_all_feat, img_feats_list[i+1].shape[2:]) # [B, next_H*next_W, C]
#                     next_to_all_feat = self.SA_blocks_1[i][j][1](next_to_all_feat, img_feats_list[i+1].shape[2:])   # [B, next_H*next_W, C]
#                     to_re_feat_inuse = self.SA_blocks_2[i][j][0](to_re_feat_inuse, (int(math.sqrt(curr_num_neighbor)), int(math.sqrt(curr_num_neighbor))))
#                     to_re_feat_inuse = self.SA_blocks_2[i][j][1](to_re_feat_inuse, (int(math.sqrt(curr_num_neighbor)), int(math.sqrt(curr_num_neighbor))))
#                     next_to_re_feat = torch.gather(next_to_all_feat,
#                                                 dim=1,
#                                                 index=next_to_re_feat_idx.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [B, num_k_2 * num_re, C]
#                     next_to_re_feat = next_to_re_feat.reshape(
#                         B, num_k_2, num_re, self.out_dim).permute(
#                             0, 2, 1, 3).reshape(
#                                 -1, num_k_2, self.out_dim) # [B * num_re, num_k_2, C]
#                     (next_to_re_feat, 
#                     attn1, 
#                     to_re_feat_inuse, 
#                     attn2) = self.CA_blocks[i][j](next_to_re_feat, to_re_feat_inuse) # [B * num_re, num_k_2, C]、
#                                                                                 # [B * num_re, num_k_2, curr_num_neighbor]、
#                                                                                 # [B * num_re, curr_num_neighbor, C]、
#                                                                                 # [B * num_re, curr_num_neighbor, num_k_2]
#                     next_to_all_feat.scatter_(dim=1, 
#                                             index=next_to_re_feat_idx.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim),  # (B, num_k_2 * num_re, C)
#                                             src=next_to_re_feat.reshape(B, num_re, num_k_2, self.out_dim).flatten(1, 2),
#                                             reduce='add') # [B, num_k_2 * num_re, C]
#                     next_to_all_feat = next_to_all_feat / next_to_all_feat_num # [B, next_H*next_W, C]
#             elif self.feat_attn_type == 'only_re_feat':
#                 next_to_re_feat = torch.gather(next_to_all_feat,
#                                             dim=1,
#                                             index=next_to_re_feat_idx.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [B, num_k_2 * num_re, C]
#                 next_to_re_feat = next_to_re_feat.reshape(
#                     B, num_k_2, num_re, self.out_dim).permute(
#                         0, 2, 1, 3).reshape(
#                             -1, num_k_2, self.out_dim) # [B * num_re, num_k_2, C]
#                 for j in range(self.depth_in_one_layer):
#                     next_to_re_feat = self.SA_blocks_1[i][j][0](next_to_re_feat, (int(math.sqrt(num_k_2)), int(math.sqrt(num_k_2)))) # [B * num_re, num_k_2, C]
#                     next_to_re_feat = self.SA_blocks_1[i][j][1](next_to_re_feat, (int(math.sqrt(num_k_2)), int(math.sqrt(num_k_2)))) # [B * num_re, num_k_2, C]
#                     to_re_feat_inuse = self.SA_blocks_2[i][j][0](to_re_feat_inuse, (int(math.sqrt(curr_num_neighbor)), int(math.sqrt(curr_num_neighbor)))) # [B * num_re, curr_num_neighbor, C]
#                     to_re_feat_inuse = self.SA_blocks_2[i][j][1](to_re_feat_inuse, (int(math.sqrt(curr_num_neighbor)), int(math.sqrt(curr_num_neighbor)))) # [B * num_re, curr_num_neighbor, C]
#                     (next_to_re_feat, 
#                     attn1, 
#                     to_re_feat_inuse, 
#                     attn2) = self.CA_blocks[i][j](next_to_re_feat, to_re_feat_inuse) # [B * num_re, num_k_2, C]、
#                                                                                 # [B * num_re, num_k_2, curr_num_neighbor]、
#                                                                                 # [B * num_re, curr_num_neighbor, C]、
#                                                                                 # [B * num_re, curr_num_neighbor, num_k_2]
#             else:
#                 raise NotImplementedError

#             attn1 = attn1.reshape(B, num_re, num_k_2, curr_num_neighbor)
#             attn2 = attn2.reshape(B, num_re, curr_num_neighbor, num_k_2)
#             if self.attn_use_type == 'mean':
#                 attn = (attn1 + attn2.permute(0, 1, 3, 2)) / 2 # [B, num_re, num_k_2, curr_num_neighbor]
#             elif self.attn_use_type == '1':
#                 attn = attn1
#             elif self.attn_use_type == '2':
#                 attn = attn2.permute(0, 1, 3, 2)
#             else:
#                 raise NotImplementedError
#             positive_mask = torch.gt(to_re_feat_attn_inuse, curr_attn_threshold[:, 0].unsqueeze(-1).unsqueeze(-1))
#             negative_mask = torch.le(to_re_feat_attn_inuse, curr_attn_threshold[:, 1].unsqueeze(-1).unsqueeze(-1))
#             to_re_feat_attn_inuse = torch.where(negative_mask, 1.0 - to_re_feat_attn_inuse, to_re_feat_attn_inuse)
#             mask = positive_mask.type(torch.float32) - negative_mask.type(torch.float32)
#             to_re_feat_attn_inuse = to_re_feat_attn_inuse * mask # [B, num_re, curr_num_neighbor]
#             attn_next = torch.einsum('b r k c, b r c -> b r k', attn, to_re_feat_attn_inuse) # [B, num_re, num_k_2]
#             attn_next_min = torch.min(attn_next.flatten(1), dim=-1, keepdim=False)[0]
#             attn_next_min = torch.where(attn_next_min <= 0.0, -attn_next_min, torch.zeros_like(attn_next_min)) + 1e-6
#             attn_next = attn_next + attn_next_min.unsqueeze(-1).unsqueeze(-1)

#             attn_next_max_in_feature = torch.max(attn_next, dim=-1, keepdim=True)[0] # [B, num_re, 1]
#             attn_next_max_in_image = torch.max(attn_next_max_in_feature, dim=-2, keepdim=True)[0] # [B, 1, 1]
#             attn_temp = attn_next / attn_next_max_in_image # [B, num_re, num_k_2]
#             attn_temp = attn_temp.reshape(B, num_re * num_k_2)
#             # if self.attn_type == 'mean':
#             #     attn_all = torch.full((B, next_H * next_W), -1e6, device=device, dtype=attn_temp.dtype)
#             #     attn_num = torch.zeros((B, next_H * next_W), device=device, dtype=attn_temp.dtype)
#             #     attn_temp_num = torch.ones_like(attn_temp)
#             #     attn_all.scatter_(dim=1, index=next_to_re_feat_idx.permute(0, 2, 1).flatten(1), src=attn_temp) # [B, next_H*next_W]
#             #     attn_num.scatter_(dim=1, index=next_to_re_feat_idx.permute(0, 2, 1).flatten(1), src=attn_temp_num) # [B, next_H*next_W]
#             #     attn_num = torch.where(attn_num == 0.0, torch.ones_like(attn_num), attn_num)
#             #     attn_all = attn_all / attn_num # [B, next_H*next_W]  the average attn score of the same pixel
#             if self.attn_type == 'mean':
#                 attn_all = torch_scatter.scatter_mean(
#                     attn_temp, 
#                     next_to_re_feat_idx.permute(0, 2, 1).flatten(1), 
#                     dim=1, 
#                     dim_size=next_H * next_W
#                     ) # [B, next_H*next_W]
#             elif self.attn_type == 'max':
#                 attn_all = torch_scatter.scatter_max(
#                     attn_temp, 
#                     next_to_re_feat_idx.permute(0, 2, 1).flatten(1), 
#                     dim=1, 
#                     dim_size=next_H * next_W
#                     )[0] # [B, next_H*next_W]
#             elif self.attn_type == 'min':
#                 attn_all = torch_scatter.scatter_min(
#                     attn_temp, 
#                     next_to_re_feat_idx.permute(0, 2, 1).flatten(1), 
#                     dim=1, 
#                     dim_size=next_H * next_W
#                     )[0]
#             else:
#                 raise NotImplementedError
#             attn_temp = torch.gather(attn_all,
#                                      dim=-1,
#                                      index=next_to_re_feat_idx_1.permute(0, 2, 1).flatten(1)) # (B, num_re * num_k_1)
#             attn_inuse_sorted_idxs_temp = torch.argsort(attn_temp, dim=-1, descending=True) # [B, num_re * num_k_1]
#             attn_inuse_sorted_idxs = torch.gather(next_to_re_feat_idx_1.permute(0, 2, 1).reshape(B, -1),
#                                                 dim=-1,
#                                                 index=attn_inuse_sorted_idxs_temp) # [B, num_re * num_k_1]
#             attn_inuse = attn_all
#             coarsest_feat = img_feats_list[i+1].flatten(2).permute(2, 0, 1) # [next_H * next_W, B, C]
#             to_aggr_feat_idx = attn_inuse_sorted_idxs[:, :int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[1])]
#             to_aggr_feat_idx_list.append(to_aggr_feat_idx)
#         return aggr_feat_list, to_aggr_feat_idx_list

    
#     @staticmethod
#     def sinusoidal_embedding(n_channels, dim):
#         pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
#                                 for p in range(n_channels)])
#         pe[:, 0::2] = torch.sin(pe[:, 0::2])
#         pe[:, 1::2] = torch.cos(pe[:, 1::2])
#         return pe.unsqueeze(0)

class ImgAttnRe(nn.Module):

    def __init__(self, cfgs, out_dim):
        super(ImgAttnRe, self).__init__()
        assert len(cfgs.high_score_ratio_list) == len(cfgs.low_score_ratio_list)
        assert len(cfgs.ca_cfgs_list) == len(cfgs.sa_cfgs_list)
        self.high_score_ratio_list = cfgs.high_score_ratio_list
        self.num_neighbor_list = cfgs.num_neighbor_list
        self.low_score_ratio_list = cfgs.low_score_ratio_list
        self.out_dim = out_dim
        self.depth_in_one_layer = cfgs.depth_in_one_layer
        self.first_SA_blocks = nn.ModuleList()
        self.SA_blocks_1 = nn.ModuleList()
        self.SA_blocks_2 = nn.ModuleList()
        self.CA_blocks = nn.ModuleList()
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, out_dim),
                                       requires_grad=True)
        self.pos_emb = nn.Parameter(self.sinusoidal_embedding(cfgs.sequence_length + 1, out_dim),
                                    requires_grad=False)
        first_SA_block = ResidualSelfAttentionBlock_v2(out_dim,
                                                       cfgs.first_SA_block_cfgs)
        self.layer_num = len(cfgs.sa_cfgs_list)
        for i in range(cfgs.first_SA_block_cfgs.num_blocks):
            self.first_SA_blocks.append(deepcopy(first_SA_block))
        for i in range(len(cfgs.sa_cfgs_list)):
            curr_resolution_SA_blocks_1 = nn.ModuleList()
            curr_resolution_SA_blocks_2 = nn.ModuleList()
            for j in range(self.depth_in_one_layer):
                curr_depth_SA_blocks_1 = nn.ModuleList()
                curr_depth_SA_blocks_2 = nn.ModuleList()
                curr_depth_SA_blocks_1.append(SwinBlock(out_dim, cfgs.sa_cfgs_list[i][0], shift=False))
                curr_depth_SA_blocks_1.append(SwinBlock(out_dim, cfgs.sa_cfgs_list[i][0], shift=True))
                curr_depth_SA_blocks_2.append(SwinBlock(out_dim, cfgs.sa_cfgs_list[i][1], shift=False))
                curr_depth_SA_blocks_2.append(SwinBlock(out_dim, cfgs.sa_cfgs_list[i][1], shift=True))
                curr_resolution_SA_blocks_1.append(curr_depth_SA_blocks_1)
                curr_resolution_SA_blocks_2.append(curr_depth_SA_blocks_2)
            self.SA_blocks_1.append(curr_resolution_SA_blocks_1)
            self.SA_blocks_2.append(curr_resolution_SA_blocks_2)
        for i in range(len(cfgs.ca_cfgs_list)):
            curr_resolution_CA_blocks = nn.ModuleList()
            for j in range(self.depth_in_one_layer):
                curr_resolution_CA_blocks.append(CwindowBlock(out_dim, cfgs.ca_cfgs_list[i]))
            self.CA_blocks.append(curr_resolution_CA_blocks)
        self.attn_type = cfgs.attn_type
        self.attn_use_type = cfgs.attn_use_type
        self.feat_attn_type = cfgs.feat_attn_type
        self.only_re_feat_type = cfgs.only_re_feat_type
        self.to_aggr_feat_select_type = cfgs.to_aggr_feat_select_type
        self.feat_fuse_type = cfgs.feat_fuse_type


    def forward(self, img_feats_list):
        B = img_feats_list[0].size(0)
        device = img_feats_list[0].device
        device_id = device.index
        aggr_feat_list = []
        if self.to_aggr_feat_select_type == 1 or self.to_aggr_feat_select_type == 2:
            to_aggr_feat_idx_list = []
            for i in range(len(img_feats_list)):
                to_aggr_feat_idx_list.append([])
        elif self.to_aggr_feat_select_type == 3:
            to_aggr_feat_list = []
        else:
            raise NotImplementedError
        coarsest_feat = img_feats_list[0].flatten(2).permute(2, 0, 1) # [B, C, H, W] -> [B, C, H*W] -> [H*W, B, C]
        assert coarsest_feat.size(0) == (self.pos_emb.size(1) - 1)
        resolution_list = [temp_feats.shape[2:] for temp_feats in img_feats_list]
        resolution_list.insert(0, (1, 1))
        resolution_list = resolution_list[::-1]
        index_list, knn_index_list, img_mesh_list = generate_img_index_and_knn_and_coords_v3(B, 
                                                                                             resolution_list, 
                                                                                             device)

        cls_token = self.cls_emb.expand(-1, coarsest_feat.size(1), -1) # [1, B, C]
        coarsest_feat = torch.cat([cls_token, coarsest_feat], dim=0) + self.pos_emb.permute(1, 0, 2) # [H*W+1, B, C]
        for i in range(len(self.first_SA_blocks)):
            coarsest_feat, attn_weights = self.first_SA_blocks[i](coarsest_feat) # [H*W + 1, B, C]、[B, H*W + 1, H*W + 1]
        

        aggr_feat_list.append(coarsest_feat[0, :, :])
        coarsest_feat = coarsest_feat[1:, :, :] # [H*W, B, C]
        attn_inuse = attn_weights[:, 0, 1:] # [B, H*W]
        attn_inuse_sorted_idxs = torch.argsort(attn_inuse, dim=1, descending=True) # [B, H*W]
        to_aggr_feat_idx = attn_inuse_sorted_idxs[:, :int(attn_inuse.size(1) * self.high_score_ratio_list[0])]
        if self.to_aggr_feat_select_type == 1 or self.to_aggr_feat_select_type == 2:
            to_aggr_feat_idx_list[0].append(to_aggr_feat_idx)
            if self.to_aggr_feat_select_type == 2:
                for i in range(1, len(img_feats_list)):
                    diff_neighbor_num_temp = knn_index_list[-(i+1)].shape[1]
                    to_aggr_feat_idx = torch.gather(knn_index_list[-(i+1)],
                                                    dim=2,
                                                    index=to_aggr_feat_idx.unsqueeze(-2).expand(-1, diff_neighbor_num_temp, -1)) 
                    to_aggr_feat_idx = to_aggr_feat_idx.flatten(1)
                    to_aggr_feat_idx_list[i].append(to_aggr_feat_idx)
        elif self.to_aggr_feat_select_type == 3:
            to_aggr_feat_temp = torch.gather(coarsest_feat, 
                                             dim=0, 
                                             index=to_aggr_feat_idx.permute(1, 0).unsqueeze(-1).expand(-1, -1, self.out_dim))
            to_aggr_feat_list.append(to_aggr_feat_temp.permute(1, 0, 2))
        
            
        for i in range(self.layer_num):
            to_re_feat_idx = attn_inuse_sorted_idxs[:, int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[i]): int(attn_inuse_sorted_idxs.size(1) * self.low_score_ratio_list[i])] # [B, num_re]
            curr_attn_threshold = []
            curr_attn_threshold.append(
                torch.gather(
                    attn_inuse,
                    dim=1,
                    index=attn_inuse_sorted_idxs[:, int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[i]):int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[i])+1]))
            curr_attn_threshold.append(
                torch.gather(
                    attn_inuse,
                    dim=1,
                    index=attn_inuse_sorted_idxs[:, int(attn_inuse_sorted_idxs.size(1) * self.low_score_ratio_list[i]):int(attn_inuse_sorted_idxs.size(1) * self.low_score_ratio_list[i])+1]))
            curr_attn_threshold = torch.cat(curr_attn_threshold, dim=1) # [B, 2]
            to_all_feat_mesh = img_mesh_list[-(i+2)]
            to_re_feat_mesh = torch.gather(to_all_feat_mesh,
                                        dim=1,
                                        index=to_re_feat_idx.unsqueeze(-1).expand(-1, -1, 2)) # [B, num_re, 2]
            to_re_feat_mesh_lazy = LazyTensor(to_re_feat_mesh.contiguous().unsqueeze(-2)) # [B, num_re, 1, 2]
            to_all_feat_mesh_lazy = LazyTensor(to_all_feat_mesh.contiguous().unsqueeze(-3)) # [B, 1, H*W, 2] 
            re_to_all_feat_mesh_dist_lazy = (to_re_feat_mesh_lazy - to_all_feat_mesh_lazy).norm2() # [B, num_re, H*W]
            _, to_re_feat_idx_inuse = re_to_all_feat_mesh_dist_lazy.Kmin_argKmin(self.num_neighbor_list[i], dim=2, device_id=device_id) # [B, num_re, curr_neighbor_num]
            to_re_feat_idx_inuse = torch.sort(to_re_feat_idx_inuse, dim=-1, descending=False)[0] # ensure the order of the index same as the spatial distribution
            num_re, curr_neighbor_num = to_re_feat_idx_inuse.size(1), to_re_feat_idx_inuse.size(2)
            to_re_feat_attn_inuse = torch.gather(attn_inuse,
                                                dim=1,
                                                index=to_re_feat_idx_inuse.flatten(1)) # [B, num_re * curr_neighbor_num]
            assert torch.count_nonzero(to_re_feat_attn_inuse == 0.) == 0
            to_re_feat_attn_inuse = to_re_feat_attn_inuse.reshape(B, -1, curr_neighbor_num) # [B, num_re, curr_neighbor_num]
            to_re_feat_inuse = torch.gather(coarsest_feat,
                                            dim=0,
                                            index=to_re_feat_idx_inuse.permute(1, 2, 0).flatten(0, 1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [num_re * curr_neighbor_num, B, C]
            to_re_feat_inuse = to_re_feat_inuse.permute(1, 0, 2).reshape(B, num_re, curr_neighbor_num, self.out_dim).reshape(-1, curr_neighbor_num, self.out_dim) # [B * num_re, curr_neighbor_num, C]
            diff_neighbor_num = knn_index_list[-(i+2)].shape[1]
            next_to_re_feat_idx = torch.gather(knn_index_list[-(i+2)],
                                                dim=-1,
                                                index=to_re_feat_idx.unsqueeze(1).expand(-1, diff_neighbor_num, -1)) # [B, diff_neighbor_num, num_re]
            next_to_re_feat_idx = next_to_re_feat_idx.permute(0, 2, 1) # [B, num_re, diff_neighbor_num]

            next_neighbor_num = self.num_neighbor_list[i+1]
            next_to_all_feat_mesh = img_mesh_list[-(i+3)] # [B, next_H*next_W, 2]
            next_to_re_feat_mesh = torch.gather(next_to_all_feat_mesh,
                                                dim=1,
                                                index=next_to_re_feat_idx.flatten(1).unsqueeze(-1).expand(-1, -1, 2)) # [B, num_re * diff_neighbor_num, 2]
            next_to_all_feat_mesh_lazy = LazyTensor(next_to_all_feat_mesh.contiguous().unsqueeze(-3)) # [B, 1, next_H*next_W, 2]
            next_to_re_feat_mesh_lazy = LazyTensor(next_to_re_feat_mesh.contiguous().unsqueeze(-2)) # [B, num_re * diff_neighbor_num, 1, 2]
            next_re_to_next_all_feat_mesh_dist_lazy = (next_to_re_feat_mesh_lazy - next_to_all_feat_mesh_lazy).norm2() # [B, num_re * diff_neighbor_num, next_H*next_W]
            next_to_re_feat_idx_inuse = next_re_to_next_all_feat_mesh_dist_lazy.Kmin_argKmin(
                K=next_neighbor_num,
                dim=2,
                device_id=device_id)[1] # [B, num_re * diff_num_neighbor, next_neighbor_num]
            next_to_re_feat_idx_inuse = torch.sort(next_to_re_feat_idx_inuse, dim=-1, descending=False)[0] # (B, num_re * diff_num_neighbor, next_neighbor_num)
            next_H, next_W = img_feats_list[i+1].shape[2:]
            next_to_all_feat = img_feats_list[i+1].flatten(2).permute(0, 2, 1) # [B, C, next_H, next_W] -> [B, C, next_H*next_W] -> [B, next_H*next_W, C]

            if self.feat_attn_type == 'all_feat':
                raise NotImplementedError
                # TODO: to fix
                # next_to_all_feat_num = torch.zeros_like(next_to_all_feat) # [B, next_H*next_W, C]
                # next_to_re_feat_num = torch.ones((B, diff_neighbor_num * next_neighbor_num, num_re, self.out_dim), dtype=next_to_all_feat_num.dtype, device=device) # [B, diff_neighbor_num * next_neighbor_num, num_re, C]
                # next_to_all_feat_num.scatter_(dim=1, 
                #                             index=next_to_re_feat_idx.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim), 
                #                             src=next_to_re_feat_num.flatten(1, 2),
                #                             reduce='add')
                # next_to_all_feat_num = torch.where(next_to_all_feat_num == 0.0, torch.ones_like(next_to_all_feat_num), next_to_all_feat_num)
                # for j in range(self.depth_in_one_layer):
                #     next_to_all_feat = self.SA_blocks_1[i][j][0](next_to_all_feat, img_feats_list[i+1].shape[2:]) # [B, next_H*next_W, C]
                #     next_to_all_feat = self.SA_blocks_1[i][j][1](next_to_all_feat, img_feats_list[i+1].shape[2:])   # [B, next_H*next_W, C]
                #     to_re_feat_inuse = self.SA_blocks_2[i][j][0](to_re_feat_inuse, (int(math.sqrt(curr_num_neighbor)), int(math.sqrt(curr_num_neighbor))))
                #     to_re_feat_inuse = self.SA_blocks_2[i][j][1](to_re_feat_inuse, (int(math.sqrt(curr_num_neighbor)), int(math.sqrt(curr_num_neighbor))))
                #     next_to_re_feat = torch.gather(next_to_all_feat,
                #                                 dim=1,
                #                                 index=next_to_re_feat_idx.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [B, num_k_2 * num_re, C]
                #     next_to_re_feat = next_to_re_feat.reshape(
                #         B, num_k_2, num_re, self.out_dim).permute(
                #             0, 2, 1, 3).reshape(
                #                 -1, num_k_2, self.out_dim) # [B * num_re, num_k_2, C]
                #     (next_to_re_feat, 
                #     attn1, 
                #     to_re_feat_inuse, 
                #     attn2) = self.CA_blocks[i][j](next_to_re_feat, to_re_feat_inuse) # [B * num_re, num_k_2, C]、
                #                                                                 # [B * num_re, num_k_2, curr_num_neighbor]、
                #                                                                 # [B * num_re, curr_num_neighbor, C]、
                #                                                                 # [B * num_re, curr_num_neighbor, num_k_2]
                #     next_to_all_feat.scatter_(dim=1, 
                #                             index=next_to_re_feat_idx.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim),  # (B, num_k_2 * num_re, C)
                #                             src=next_to_re_feat.reshape(B, num_re, num_k_2, self.out_dim).flatten(1, 2),
                #                             reduce='add') # [B, num_k_2 * num_re, C]
                #     next_to_all_feat = next_to_all_feat / next_to_all_feat_num # [B, next_H*next_W, C]
            elif self.feat_attn_type == 'only_re_feat':
                next_to_re_feat_idx_inuse_temp = next_to_re_feat_idx_inuse.reshape(B, num_re, diff_neighbor_num, next_neighbor_num) # [B, num_re, diff_neighbor_num, next_neighbor_num]
                next_to_re_feat_idx_inuse_temp = next_to_re_feat_idx_inuse_temp.reshape(B, num_re, diff_neighbor_num * next_neighbor_num) # [B, num_re, diff_neighbor_num * next_neighbor_num]
                next_to_re_feat_idx_inuse_temp_sorted, next_to_re_feat_idx_inuse_temp_sorted_idx = torch.sort(next_to_re_feat_idx_inuse_temp, dim=-1, descending=False) # [B, num_re, diff_neighbor_num * next_neighbor_num]
                to_reverse_idx = torch.arange(diff_neighbor_num * next_neighbor_num, device=device).unsqueeze(0).unsqueeze(0).expand(B, num_re, -1) # [B, num_re, diff_neighbor_num * next_neighbor_num]
                next_to_re_feat_idx_inuse_temp_sorted_idx_reverse = torch_scatter.scatter_sum(src=to_reverse_idx,
                                                                                                index=next_to_re_feat_idx_inuse_temp_sorted_idx,
                                                                                                dim=-1,
                                                                                                dim_size=diff_neighbor_num * next_neighbor_num) # [B, num_re, diff_neighbor_num * next_neighbor_num]
                if self.only_re_feat_type == 1:
                    next_to_re_feat_inuse_temp_sorted = torch.gather(next_to_all_feat, 
                                                                        dim=1, 
                                                                        index=next_to_re_feat_idx_inuse_temp_sorted.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [B, num_re * diff_neighbor_num * next_neighbor_num, C]
                    next_to_re_feat_inuse_temp_sorted = next_to_re_feat_inuse_temp_sorted.reshape(B, num_re, diff_neighbor_num, next_neighbor_num, self.out_dim) # [B, num_re, diff_neighbor_num, next_neighbor_num, C]
                    next_to_re_feat_inuse_sorted = next_to_re_feat_inuse_temp_sorted.reshape(B * num_re, diff_neighbor_num * next_neighbor_num, self.out_dim) # [B * num_re, diff_neighbor_num * next_neighbor_num, C]
                    for j in range(self.depth_in_one_layer):
                        next_to_re_feat_inuse_sorted = self.SA_blocks_1[i][j][0](next_to_re_feat_inuse_sorted, (int(math.sqrt(diff_neighbor_num * next_neighbor_num)), int(math.sqrt(diff_neighbor_num * next_neighbor_num)))) # [B * num_re, diff_neighbor_num * next_neighbor_num, C]
                        next_to_re_feat_inuse_sorted = self.SA_blocks_1[i][j][1](next_to_re_feat_inuse_sorted, (int(math.sqrt(diff_neighbor_num * next_neighbor_num)), int(math.sqrt(diff_neighbor_num * next_neighbor_num)))) # [B * num_re, diff_neighbor_num * next_neighbor_num, C]
                        to_re_feat_inuse = self.SA_blocks_2[i][j][0](to_re_feat_inuse, (int(math.sqrt(curr_neighbor_num)), int(math.sqrt(curr_neighbor_num)))) # [B * num_re, curr_neighbor_num, C]
                        to_re_feat_inuse = self.SA_blocks_2[i][j][1](to_re_feat_inuse, (int(math.sqrt(curr_neighbor_num)), int(math.sqrt(curr_neighbor_num)))) # [B * num_re, curr_neighbor_num, C]
                        # next_to_re_feat_inuse = torch.gather(input=next_to_re_feat_inuse_sorted, dim=-1, index=next_to_re_feat_idx_inuse_temp_sorted_idx_reverse.flatten(0, 1).unsqueeze()) # [B * num_re, diff_neighbor_num * next_neighbor_num, C]
                        (next_to_re_feat_inuse_sorted, 
                        attn1, 
                        to_re_feat_inuse, 
                        attn2) = self.CA_blocks[i][j](next_to_re_feat_inuse_sorted, to_re_feat_inuse) # [B * num_re, diff_neighbor_num * next_neighbor_num, C]、
                                                                                    # [B * num_re, diff_neighbor_num * next_neighbor_num, curr_neighbor_num]、
                                                                                    # [B * num_re, curr_neighbor_num, C]、
                                                                                    # [B * num_re, curr_neighbor_num, diff_neighbor_num * next_neighbor_num]
                    attn2 = attn2.permute(0, 2, 1)
                    attn1 = torch.gather(attn1, 
                                         dim=1, 
                                         index=next_to_re_feat_idx_inuse_temp_sorted_idx_reverse.flatten(0, 1).unsqueeze(-1).expand(-1, -1, curr_neighbor_num)) # [B * num_re, diff_neighbor_num * next_neighbor_num, curr_neighbor_num]
                    attn2 = torch.gather(attn2,
                                         dim=1,
                                         index=next_to_re_feat_idx_inuse_temp_sorted_idx_reverse.flatten(0, 1).unsqueeze(-1).expand(-1, -1, curr_neighbor_num)) # [B * num_re, diff_neighbor_num * next_neighbor_num, curr_neighbor_num]
                    attn1 = attn1.reshape(B, num_re, diff_neighbor_num, next_neighbor_num, curr_neighbor_num)
                    attn2 = attn2.reshape(B, num_re, diff_neighbor_num, next_neighbor_num, curr_neighbor_num)
                elif self.only_re_feat_type == 2:
                    next_to_re_feat_inuse = torch.gather(next_to_all_feat,
                                            dim=1,
                                            index=next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [B, num_re * diff_num_neighbor * next_neighbor_num, C]
                    next_to_re_feat_inuse = next_to_re_feat_inuse.reshape(B, num_re, diff_neighbor_num, next_neighbor_num, self.out_dim) # [B, num_re, diff_neighbor_num, next_neighbor_num, C]
                    next_to_re_feat_inuse = next_to_re_feat_inuse.reshape(B * num_re * diff_neighbor_num, next_neighbor_num, self.out_dim) # [B * num_re * diff_neighbor_num, next_neighbor_num, C]
                    for j in range(self.depth_in_one_layer):
                        next_to_re_feat_inuse = self.SA_blocks_1[i][j][0](next_to_re_feat_inuse, (int(math.sqrt(next_neighbor_num)), int(math.sqrt(next_neighbor_num)))) # [B * num_re * diff_neighbor_num, next_neighbor_num, C]
                        next_to_re_feat_inuse = self.SA_blocks_1[i][j][1](next_to_re_feat_inuse, (int(math.sqrt(next_neighbor_num)), int(math.sqrt(next_neighbor_num)))) # [B * num_re * diff_neighbor_num, next_neighbor_num, C]
                        to_re_feat_inuse = self.SA_blocks_2[i][j][0](to_re_feat_inuse, (int(math.sqrt(curr_neighbor_num)), int(math.sqrt(curr_neighbor_num)))) # [B * num_re, curr_neighbor_num, C]
                        to_re_feat_inuse = self.SA_blocks_2[i][j][1](to_re_feat_inuse, (int(math.sqrt(curr_neighbor_num)), int(math.sqrt(curr_neighbor_num)))) # [B * num_re, curr_neighbor_num, C]
                        next_to_re_feat_inuse = next_to_re_feat_inuse.reshape(B, num_re, diff_neighbor_num, next_neighbor_num, self.out_dim) # [B, num_re, diff_neighbor_num, next_neighbor_num, C]
                        next_to_re_feat_inuse = next_to_re_feat_inuse.reshape(B * num_re, diff_neighbor_num * next_neighbor_num, self.out_dim) # [B * num_re, diff_neighbor_num * next_neighbor_num, C]
                        next_to_re_feat_inuse_sorted = torch.gather(next_to_re_feat_inuse,
                                                                    dim=1,
                                                                    index=next_to_re_feat_idx_inuse_temp_sorted_idx.flatten(0, 1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [B * num_re, diff_neighbor_num * next_neighbor_num, C]
                        (next_to_re_feat_inuse_sorted, 
                        attn1, 
                        to_re_feat_inuse, 
                        attn2) = self.CA_blocks[i][j](next_to_re_feat_inuse_sorted, to_re_feat_inuse) # [B * num_re, diff_neighbor_num * next_neighbor_num, C]、
                                                                                    # [B * num_re, diff_neighbor_num * next_neighbor_num, curr_neighbor_num]、
                                                                                    # [B * num_re, curr_neighbor_num, C]、
                                                                                    # [B * num_re, curr_neighbor_num, diff_neighbor_num * next_neighbor_num]
                        next_to_re_feat_inuse = torch.gather(next_to_re_feat_inuse_sorted,
                                                             dim=1,
                                                             index=next_to_re_feat_idx_inuse_temp_sorted_idx_reverse.flatten(0, 1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [B * num_re, diff_neighbor_num * next_neighbor_num, C]
                    attn2 = attn2.permute(0, 2, 1)
                    attn1 = torch.gather(attn1, 
                                         dim=1, 
                                         index=next_to_re_feat_idx_inuse_temp_sorted_idx_reverse.flatten(0, 1).unsqueeze(-1).expand(-1, -1, curr_neighbor_num)) # [B * num_re, diff_neighbor_num * next_neighbor_num, curr_neighbor_num]
                    attn2 = torch.gather(attn2,
                                         dim=1,
                                         index=next_to_re_feat_idx_inuse_temp_sorted_idx_reverse.flatten(0, 1).unsqueeze(-1).expand(-1, -1, curr_neighbor_num)) # [B * num_re, diff_neighbor_num * next_neighbor_num, curr_neighbor_num]
                    attn1 = attn1.reshape(B, num_re, diff_neighbor_num, next_neighbor_num, curr_neighbor_num)
                    attn2 = attn2.reshape(B, num_re, diff_neighbor_num, next_neighbor_num, curr_neighbor_num)
            else:
                raise NotImplementedError

            if self.attn_use_type == 'mean':
                attn = (attn1 + attn2) / 2 # [B, num_re, diff_neighbor_num, next_neighbor_num, curr_neighbor_num]
            elif self.attn_use_type == '1':
                attn = attn1 # [B, num_re, diff_neighbor_num, next_neighbor_num, curr_neighbor_num]
            elif self.attn_use_type == '2':
                attn = attn2 # [B, num_re, diff_neighbor_num, next_neighbor_num, curr_neighbor_num]
            else:
                raise NotImplementedError
            positive_mask = torch.gt(to_re_feat_attn_inuse, curr_attn_threshold[:, 0].unsqueeze(-1).unsqueeze(-1))
            negative_mask = torch.le(to_re_feat_attn_inuse, curr_attn_threshold[:, 1].unsqueeze(-1).unsqueeze(-1))
            to_re_feat_attn_inuse = torch.where(negative_mask, 1.0 - to_re_feat_attn_inuse, to_re_feat_attn_inuse)
            mask = positive_mask.type(torch.float32) - negative_mask.type(torch.float32)
            to_re_feat_attn_inuse = to_re_feat_attn_inuse * mask # [B, num_re, curr_num_neighbor]
            attn_next = torch.einsum('b r d n c, b r c -> b r d n', attn, to_re_feat_attn_inuse) # [B, num_re, diff_neighbor_num, next_neighbor_num]
            attn_next = attn_next.flatten(1) # [B, num_re * diff_neighbor_num * next_neighbor_num]
            attn_next_min = torch.min(attn_next, dim=-1, keepdim=False)[0]
            attn_next_min = torch.where(attn_next_min <= 0.0, -attn_next_min, torch.zeros_like(attn_next_min)) + 1e-6
            attn_next = attn_next + attn_next_min.unsqueeze(-1)
            attn_next_max_in_image = torch.max(attn_next, dim=-1, keepdim=True)[0] # [B, 1]
            attn_next = attn_next / attn_next_max_in_image # [B, num_re * diff_neighbor_num * next_neighbor_num]
            attn_next = torch.clamp(attn_next, min=1e-6)
            if self.attn_type == 'mean':
                attn_all = torch_scatter.scatter_mean(
                    attn_next, 
                    next_to_re_feat_idx_inuse.flatten(1), 
                    dim=1, 
                    dim_size=next_H * next_W
                    ) # [B, next_H*next_W]
            elif self.attn_type == 'max':
                attn_all = torch_scatter.scatter_max(
                    attn_next, 
                    next_to_re_feat_idx_inuse.flatten(1), 
                    dim=1, 
                    dim_size=next_H * next_W
                    )[0] # [B, next_H*next_W]
            elif self.attn_type == 'min':
                attn_all = torch_scatter.scatter_min(
                    attn_next, 
                    next_to_re_feat_idx_inuse.flatten(1), 
                    dim=1, 
                    dim_size=next_H * next_W
                    )[0] # (B, next_H*next_W)
            else:
                raise NotImplementedError
            attn_temp = torch.gather(attn_all,
                                     dim=-1,
                                     index=next_to_re_feat_idx.flatten(1)) # (B, num_re * diff_num_neighbor)
            attn_inuse_sorted_idxs_temp = torch.argsort(attn_temp, dim=-1, descending=True) # [B, num_re * diff_neighbor_num]
            attn_inuse_sorted_idxs = torch.gather(next_to_re_feat_idx.flatten(1),
                                                dim=-1,
                                                index=attn_inuse_sorted_idxs_temp) # [B, num_re * diff_neighbor_num]
            attn_inuse = attn_all
            coarsest_feat = img_feats_list[i+1].flatten(2).permute(2, 0, 1) # [next_H * next_W, B, C]
            to_aggr_feat_idx = attn_inuse_sorted_idxs[:, :int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[i+1])] # [B, next_num_re]
            if self.to_aggr_feat_select_type == 1 or self.to_aggr_feat_select_type == 2:
                to_aggr_feat_idx_list[i+1].append(to_aggr_feat_idx)
                if self.to_aggr_feat_select_type == 2:
                    for j in range(i+2, len(img_feats_list)):
                        diff_neighbor_num_temp = knn_index_list[-(j+1)].shape[1]
                        to_aggr_feat_idx = torch.gather(knn_index_list[-(j+1)],
                                                        dim=2,
                                                        index=to_aggr_feat_idx.unsqueeze(-2).expand(-1, diff_neighbor_num_temp, -1))
                        to_aggr_feat_idx = to_aggr_feat_idx.flatten(1)
                        to_aggr_feat_idx_list[j].append(to_aggr_feat_idx)
            elif self.to_aggr_feat_select_type == 3:
                if self.only_re_feat_type == 1:
                    next_to_re_feat_inuse = torch.gather(next_to_re_feat_inuse_sorted,
                                                        dim=1,
                                                        index=next_to_re_feat_idx_inuse_temp_sorted_idx_reverse.flatten(0, 1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [B * num_re, diff_neighbor_num * next_neighbor_num, C]
                next_to_re_feat_inuse_temp = next_to_re_feat_inuse.reshape(B, num_re, diff_neighbor_num, next_neighbor_num, self.out_dim)
                next_to_re_feat_inuse_temp = next_to_re_feat_inuse_temp.reshape(B, num_re * diff_neighbor_num * next_neighbor_num, self.out_dim)
                if self.feat_fuse_type == 'max':
                    next_feats_all_temp = torch_scatter.scatter_max(
                        next_to_re_feat_inuse_temp, 
                        next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim), 
                        dim=1, 
                        dim_size=next_H * next_W
                        )[0] # [B, next_N, C]
                elif self.feat_fuse_type == 'min':
                    next_feats_all_temp = torch_scatter.scatter_min(
                        next_to_re_feat_inuse_temp, 
                        next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim), 
                        dim=1, 
                        dim_size=next_H * next_W
                        )[0] # [B, next_N, C]
                elif self.feat_fuse_type == 'mean':
                    next_feats_all_temp = torch_scatter.scatter_mean(
                        next_to_re_feat_inuse_temp, 
                        next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim), 
                        dim=1, 
                        dim_size=next_H * next_W
                        )
                to_aggr_feat_temp = torch.gather(next_feats_all_temp,
                                                dim=1,
                                                index=next_to_re_feat_idx.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim))
                to_aggr_feat_list.append(to_aggr_feat_temp)
        
        if self.to_aggr_feat_select_type == 1 or self.to_aggr_feat_select_type == 2:
            return aggr_feat_list, to_aggr_feat_idx_list
        elif self.to_aggr_feat_select_type == 3:
            return aggr_feat_list, to_aggr_feat_list
    
    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


# class PcAttnRe(nn.Module):

#     def __init__(self, cfgs, out_dim):
#         super(PcAttnRe, self).__init__()
#         assert len(cfgs.high_score_ratio_list) == len(cfgs.low_score_ratio_list)
#         assert len(cfgs.ca_cfgs_list) == len(cfgs.sa_cfgs_list)
#         self.high_score_ratio_list = cfgs.high_score_ratio_list
#         self.num_neighbor_list = cfgs.num_neighbor_list
#         self.low_score_ratio_list = cfgs.low_score_ratio_list
#         self.out_dim = out_dim
#         self.depth_in_one_layer = cfgs.depth_in_one_layer
#         self.first_SA_blocks = nn.ModuleList()
#         self.SA_blocks_1 = nn.ModuleList()
#         self.SA_blocks_2 = nn.ModuleList()
#         self.CA_blocks = nn.ModuleList()
#         self.cls_emb = nn.Parameter(torch.zeros(1, 1, out_dim),
#                                        requires_grad=True)
#         self.use_pos_emb = cfgs.use_pos_emb
#         if self.use_pos_emb:
#             self.pos_emb = nn.Parameter(self.sinusoidal_embedding(cfgs.sequence_length + 1, out_dim),
#                                         requires_grad=False)
#         first_SA_block = ResidualSelfAttentionBlock_v2(out_dim,
#                                                        cfgs.first_SA_block_cfgs)
#         self.layer_num = len(cfgs.sa_cfgs_list)
#         for i in range(cfgs.first_SA_block_cfgs.num_blocks):
#             self.first_SA_blocks.append(deepcopy(first_SA_block))
#         for i in range(len(cfgs.sa_cfgs_list)):
#             curr_resolution_SA_blocks_1 = nn.ModuleList()
#             curr_resolution_SA_blocks_2 = nn.ModuleList()
#             for j in range(self.depth_in_one_layer):
#                 curr_resolution_SA_blocks_1.append(SABlockSeq(cfgs.sa_cfgs_list[i][0], self.out_dim, depth=self.depth_in_one_layer))
#                 curr_resolution_SA_blocks_2.append(SABlockSeq(cfgs.sa_cfgs_list[i][1], self.out_dim, depth=self.depth_in_one_layer))
#             self.SA_blocks_1.append(curr_resolution_SA_blocks_1)
#             self.SA_blocks_2.append(curr_resolution_SA_blocks_2)
#         for i in range(len(cfgs.ca_cfgs_list)):
#             curr_resolution_CA_blocks = nn.ModuleList()
#             for j in range(self.depth_in_one_layer):
#                 curr_resolution_CA_blocks.append(CABlockSeq(cfgs.ca_cfgs_list[i], self.out_dim, depth=self.depth_in_one_layer))
#             self.CA_blocks.append(curr_resolution_CA_blocks)
    
#     def forward(self, pc_feats_list, pc_coords_list):
#         B = pc_feats_list[0].size(0)
#         device = pc_feats_list[0].device
#         aggr_feat_list = []
#         to_aggr_feat_idx_list = []
#         coarsest_feat = pc_feats_list[0].permute(2, 0, 1) # [B, C, N] -> [N, B, C]
#         if self.use_pos_emb:
#             assert coarsest_feat.size(0) == (self.pos_emb.size(1) - 1)
#         pc_coords_list = pc_coords_list[::-1]
#         index_list, pc_coords_list = generate_pc_index_and_coords_v2(pc_coords_list)



#         cls_token = self.cls_emb.expand(-1, coarsest_feat.size(1), -1) # [1, B, C]
#         coarsest_feat = torch.cat([cls_token, coarsest_feat], dim=0) # [N+1, B, C]
#         if self.use_pos_emb:
#             coarsest_feat += self.pos_emb.permute(1, 0, 2) # [N+1, B, C]
#         for i in range(len(self.first_SA_blocks)):
#             coarsest_feat, attn_weights = self.first_SA_blocks[i](coarsest_feat) # [N + 1, B, C]、[B, N + 1, N + 1]
        

#         aggr_feat_list.append(coarsest_feat[0, :, :])
#         coarsest_feat = coarsest_feat[1:, :, :] # [N, B, C]
#         attn_inuse = attn_weights[:, 0, 1:] # [B, N]
#         attn_inuse_sorted_idxs = torch.argsort(attn_inuse, dim=1, descending=True) # [B, N]
#         to_aggr_feat_idx = attn_inuse_sorted_idxs[:, :int(attn_inuse.size(1) * self.high_score_ratio_list[0])]
#         to_aggr_feat_idx_list.append(to_aggr_feat_idx)


#         for i in range(self.layer_num):
#             to_re_feat_idx = attn_inuse_sorted_idxs[:, int(attn_inuse.size(1) * self.high_score_ratio_list[i]): int(attn_inuse.size(1) * self.low_score_ratio_list[i])] # [B, num_re]
#             curr_attn_threshold = []
#             curr_attn_threshold.append(
#                 torch.gather(
#                     attn_inuse,
#                     dim=1,
#                     index=attn_inuse_sorted_idxs[:, int(attn_inuse.size(1) * self.high_score_ratio_list[i]):int(attn_inuse.size(1) * self.high_score_ratio_list[i])+1]))
#             curr_attn_threshold.append(
#                 torch.gather(
#                     attn_inuse,
#                     dim=1,
#                     index=attn_inuse_sorted_idxs[:, int(attn_inuse.size(1) * self.low_score_ratio_list[i]):int(attn_inuse.size(1) * self.low_score_ratio_list[i])+1]))
#             curr_attn_threshold = torch.cat(curr_attn_threshold, dim=1) # [B, 2]
#             to_all_feat_coords = pc_coords_list[-(i+2)]
#             to_re_feat_coords = torch.gather(to_all_feat_coords,
#                                         dim=1,
#                                         index=to_re_feat_idx.unsqueeze(-1).expand(-1, -1, 3)) # [B, num_re, 3]
#             re_to_all_feat_coords_dist = torch.cdist(to_re_feat_coords, to_all_feat_coords, p=2.0) # [B, num_re, N]
#             to_re_feat_idx_inuse = torch.topk(re_to_all_feat_coords_dist, 
#                                                 k=self.num_neighbor_list[i], 
#                                                 dim=-1, 
#                                                 largest=False, 
#                                                 sorted=False)[1] # [B, num_re, curr_num_neighbor]
#             num_re, curr_num_neighbor = to_re_feat_idx_inuse.size(1), to_re_feat_idx_inuse.size(2)
#             to_re_feat_attn_inuse = torch.gather(attn_inuse,
#                                                 dim=1,
#                                                 index=to_re_feat_idx_inuse.flatten(1)) # [B, num_re * curr_num_neighbor]
#             to_re_feat_attn_inuse = to_re_feat_attn_inuse.reshape(B, -1, curr_num_neighbor) # [B, num_re, curr_num_neighbor]
#             to_re_feat_inuse = torch.gather(coarsest_feat,
#                                             dim=0,
#                                             index=to_re_feat_idx_inuse.permute(1, 2, 0).flatten(0, 1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [num_re * curr_num_neighbor, B, C]
#             to_re_feat_inuse = to_re_feat_inuse.permute(1, 0, 2).reshape(B, num_re, curr_num_neighbor, self.out_dim) # [B, num_re, curr_num_neighbor, C]
#             to_re_feat_coords_inuse = torch.gather(to_all_feat_coords,
#                                                 dim=1,
#                                                 index=to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, 3)) # [B, num_re * curr_num_neighbor, 3]
#             to_re_feat_coords_inuse = to_re_feat_coords_inuse.reshape(B, num_re, curr_num_neighbor, 3) # [B, num_re, curr_num_neighbor, 3]
#             next_to_all_feat = pc_feats_list[i+1].permute(0, 2, 1) # [B, C, next_N] -> [B, next_N, C]
#             next_to_all_index = index_list[-(i+3)] # [B, next_N]
#             to_new_re_feat_idx = inverse_mapping(to_re_feat_idx, attn_inuse_sorted_idxs.size(1)) # [B, N]
#             next_to_re_feat_idx = torch.gather(to_new_re_feat_idx,
#                                                dim=1,
#                                                index=next_to_all_index) # [B, next_N]
#             next_all_to_next_all_coords_dist = torch.cdist(pc_coords_list[-(i+3)], pc_coords_list[-(i+3)], p=2.0)


#             _, index = torch.topk(next_all_to_next_all_coords_dist, self.num_neighbor_list[i+1], dim=-1, largest=False) # (B, next_N, next_num_neighbors)
#             next_num_neighbors = index.size(-1)
#             next_to_re_feat_idx.scatter_(dim=-1, index=index.flatten(1), src=torch.repeat_interleave(next_to_re_feat_idx, next_num_neighbors, dim=-1)) # [B, next_N]

#             for j in range(self.depth_in_one_layer):
#                 next_to_all_feat = self.SA_blocks_1[i][j](next_to_all_feat, pc_coords_list[-(i+3)])
#                 to_re_feat_inuse = self.SA_blocks_2[i][j](
#                     to_re_feat_inuse.reshape(-1, curr_num_neighbor, self.out_dim), 
#                     to_re_feat_coords_inuse.reshape(-1, curr_num_neighbor, 3))
#                 to_re_feat_inuse = to_re_feat_inuse.reshape(B, num_re, curr_num_neighbor, self.out_dim)
#                 to_cat_re_feat_inuse = torch.zeros_like(to_re_feat_inuse[:, 0:1, :, :]) # [B, 1, curr_num_neighbor, C] 
#                 to_re_feat_inuse = torch.cat([to_re_feat_inuse, to_cat_re_feat_inuse], dim=1) # [B, num_re+1, curr_num_neighbor, C]
#                 to_re_feat_coords_inuse = to_re_feat_coords_inuse.reshape(B, num_re, curr_num_neighbor, 3)
#                 to_cat_re_feat_coords_inuse = torch.zeros_like(to_re_feat_coords_inuse[:, 0:1, :, :]) # [B, 1, curr_num_neighbor, 3]
#                 to_re_feat_coords_inuse = torch.cat([to_re_feat_coords_inuse, to_cat_re_feat_coords_inuse], dim=1) # [B, num_re+1, curr_num_neighbor, 3]
#                 (next_to_all_feat, 
#                 attn1, 
#                 to_re_feat_inuse, 
#                 attn2) = self.CA_blocks[i][j](next_to_all_feat, 
#                                               pc_coords_list[-(i+3)], 
#                                               to_re_feat_inuse,
#                                               to_re_feat_coords_inuse,
#                                               next_to_re_feat_idx) # [B, next_N, C]、[B, curr_num_neighbor, next_N]、[B, num_re+1, curr_num_neighbor, C]、[B, curr_num_neighbor, next_N]
#                 to_re_feat_inuse = to_re_feat_inuse[:, 1:, :, :] # [B, num_re, curr_num_neighbor, C]
#                 to_re_feat_coords_inuse = to_re_feat_coords_inuse[:, 1:, :, :]

#             attn = (attn1 + attn2) / 2 # [B, curr_num_neighbor, next_N]
#             attn = attn.permute(0, 2, 1) # [B, next_N, curr_num_neighbor]
#             to_cat_re_feat_attn_inuse = torch.zeros_like(to_re_feat_attn_inuse[:, 0:1, :]) # [B, 1, curr_num_neighbor]
#             to_re_feat_attn_inuse = torch.cat([to_re_feat_attn_inuse, to_cat_re_feat_attn_inuse], dim=1) # [B, num_re+1, curr_num_neighbor]
#             to_re_feat_attn_inuse = torch.gather(to_re_feat_attn_inuse,
#                                                  dim=1,
#                                                  index=next_to_re_feat_idx.unsqueeze(-1).expand(-1, -1, curr_num_neighbor)) # [B, next_N, curr_num_neighbor]
#             positive_mask = torch.gt(to_re_feat_attn_inuse, curr_attn_threshold[:, 0].unsqueeze(-1).unsqueeze(-1)) # [B, next_N, curr_num_neighbor]
#             negative_mask = torch.le(to_re_feat_attn_inuse, curr_attn_threshold[:, 1].unsqueeze(-1).unsqueeze(-1)) # [B, next_N, curr_num_neighbor]
#             to_re_feat_attn_inuse = torch.where(negative_mask, 1.0 - to_re_feat_attn_inuse, to_re_feat_attn_inuse) # [B, next_N, curr_num_neighbor]
#             mask = positive_mask.type(torch.float32) - negative_mask.type(torch.float32)
#             to_re_feat_attn_inuse = to_re_feat_attn_inuse * mask # [B, next_N, curr_num_neighbor]
#             attn_next = attn * to_re_feat_attn_inuse # [B, next_N, curr_num_neighbor]
#             attn_next = torch.sum(attn_next, dim=-1, keepdim=False) # [B, next_N]
#             attn_next = attn_next.masked_fill(next_to_re_feat_idx == num_re, 0.0)
#             attn_next_min = torch.min(attn_next, dim=-1)[0]
#             attn_next_min = torch.where(attn_next_min <= 0.0, -attn_next_min, torch.zeros_like(attn_next_min)) + 1e-6
#             attn_next = attn_next + attn_next_min.unsqueeze(-1)
#             attn_next_max_in_pc = torch.max(attn_next, dim=-1, keepdim=True)[0] # [B, 1]
#             attn_next = attn_next / attn_next_max_in_pc # [B, next_N]
#             attn_next = attn_next.masked_fill(next_to_re_feat_idx == num_re, -1e6)
#             attn_inuse_temp = torch.gather(attn_next, 
#                                       dim=-1, 
#                                       index=next_to_re_feat_idx)
#             attn_inuse_sorted_idxs_fake = torch.argsort(attn_inuse, dim=-1, descending=True)
#             attn_inuse_sorted_idxs = torch.gather(next_to_re_feat_idx,
#                                                 dim=-1,
#                                                 index=attn_inuse_sorted_idxs_fake) # [B, next_N]
#             coarsest_feat = pc_feats_list[i+1].permute(2, 0, 1) # [next_N, B, C]
#             to_aggr_feat_idx = attn_inuse_sorted_idxs[:, :int(attn_inuse.size(1) * self.high_score_ratio_list[1])]
#             to_aggr_feat_idx_list.append(to_aggr_feat_idx)

#         return aggr_feat_list, to_aggr_feat_idx_list

    
#     @staticmethod
#     def sinusoidal_embedding(n_channels, dim):
#         pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
#                                 for p in range(n_channels)])
#         pe[:, 0::2] = torch.sin(pe[:, 0::2])
#         pe[:, 1::2] = torch.cos(pe[:, 1::2])
#         return pe.unsqueeze(0)

class PcAttnRe(nn.Module):

    def __init__(self, cfgs, out_dim):
        super(PcAttnRe, self).__init__()
        assert len(cfgs.high_score_ratio_list) == len(cfgs.low_score_ratio_list)
        assert len(cfgs.ca_cfgs_list) == len(cfgs.sa_cfgs_list)
        self.high_score_ratio_list = cfgs.high_score_ratio_list
        self.same_res_num_neighbor_list = cfgs.same_res_num_neighbor_list
        self.diff_res_num_neighbor_list = cfgs.diff_res_num_neighbor_list
        self.low_score_ratio_list = cfgs.low_score_ratio_list
        self.out_dim = out_dim
        self.depth_in_one_layer = cfgs.depth_in_one_layer
        self.first_SA_blocks = nn.ModuleList()
        self.SA_blocks_1 = nn.ModuleList()
        self.SA_blocks_2 = nn.ModuleList()
        self.CA_blocks = nn.ModuleList()
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, out_dim),
                                       requires_grad=True)
        self.use_pos_emb = cfgs.use_pos_emb
        if self.use_pos_emb:
            self.pos_emb = nn.Parameter(self.sinusoidal_embedding(cfgs.sequence_length + 1, out_dim),
                                        requires_grad=False)
        first_SA_block = ResidualSelfAttentionBlock_v2(out_dim,
                                                       cfgs.first_SA_block_cfgs)
        self.layer_num = len(cfgs.sa_cfgs_list)
        for i in range(cfgs.first_SA_block_cfgs.num_blocks):
            self.first_SA_blocks.append(deepcopy(first_SA_block))
        for i in range(len(cfgs.sa_cfgs_list)):
            curr_resolution_SA_blocks_1 = nn.ModuleList()
            curr_resolution_SA_blocks_2 = nn.ModuleList()
            for j in range(self.depth_in_one_layer):
                curr_resolution_SA_blocks_1.append(SABlockSeq(cfgs.sa_cfgs_list[i][0], self.out_dim, depth=self.depth_in_one_layer))
                curr_resolution_SA_blocks_2.append(SABlockSeq(cfgs.sa_cfgs_list[i][1], self.out_dim, depth=self.depth_in_one_layer))
            self.SA_blocks_1.append(curr_resolution_SA_blocks_1)
            self.SA_blocks_2.append(curr_resolution_SA_blocks_2)
        for i in range(len(cfgs.ca_cfgs_list)):
            curr_resolution_CA_blocks = nn.ModuleList()
            for j in range(self.depth_in_one_layer):
                curr_resolution_CA_blocks.append(CABlockSeq(cfgs.ca_cfgs_list[i], self.out_dim, depth=self.depth_in_one_layer))
            self.CA_blocks.append(curr_resolution_CA_blocks)
        self.attn_type = cfgs.attn_type
        self.attn_use_type = cfgs.attn_use_type
        self.feat_attn_type = cfgs.feat_attn_type
        self.to_aggr_feat_select_type = cfgs.to_aggr_feat_select_type
        self.feat_fuse_type = cfgs.feat_fuse_type
    
    def forward(self, pc_feats_list, pc_coords_list):

        t0 = time.perf_counter()

        B = pc_feats_list[0].size(0)
        device = pc_feats_list[0].device
        device_id = device.index
        aggr_feat_list = []
        if self.to_aggr_feat_select_type == 1 or self.to_aggr_feat_select_type == 2:
            to_aggr_feat_idx_list = []
            for i in range(len(pc_feats_list)):
                to_aggr_feat_idx_list.append([])
        elif self.to_aggr_feat_select_type == 3:
            to_aggr_feat_list = []
        else:
            raise NotImplementedError
        coarsest_feat = pc_feats_list[0].permute(2, 0, 1) # [B, C, N] -> [N, B, C]
        if self.use_pos_emb:
            assert coarsest_feat.size(0) == (self.pos_emb.size(1) - 1)
        pc_coords_list = pc_coords_list[::-1]
        diff_res_num_neighbor_list = self.diff_res_num_neighbor_list[::-1]
        index_list, pc_coords_list = generate_pc_index_and_coords_v3(pc_coords_list, diff_res_num_neighbor_list)
        cls_token = self.cls_emb.expand(-1, coarsest_feat.size(1), -1) # [1, B, C]
        coarsest_feat = torch.cat([cls_token, coarsest_feat], dim=0) # [N+1, B, C]
        if self.use_pos_emb:
            coarsest_feat += self.pos_emb.permute(1, 0, 2) # [N+1, B, C]
        for i in range(len(self.first_SA_blocks)):
            coarsest_feat, attn_weights = self.first_SA_blocks[i](coarsest_feat) # [N + 1, B, C]、[B, N + 1, N + 1]
        aggr_feat_list.append(coarsest_feat[0, :, :])
        coarsest_feat = coarsest_feat[1:, :, :] # [N, B, C]
        attn_inuse = attn_weights[:, 0, 1:] # [B, N]
        attn_inuse_sorted_idxs = torch.argsort(attn_inuse, dim=1, descending=True) # [B, N]
        to_aggr_feat_idx = attn_inuse_sorted_idxs[:, :int(attn_inuse.size(1) * self.high_score_ratio_list[0])]
        if self.to_aggr_feat_select_type == 1 or self.to_aggr_feat_select_type == 2:
            to_aggr_feat_idx_list[0].append(to_aggr_feat_idx)
            if self.to_aggr_feat_select_type == 2:
                for i in range(1, len(pc_feats_list)):
                    diff_neighbor_num_temp = index_list[-(i+1)].shape[2]
                    to_aggr_feat_idx = torch.gather(index_list[-(i+1)],
                                                    dim=1,
                                                    index=to_aggr_feat_idx.unsqueeze(-1).expand(-1, -1, diff_neighbor_num_temp))
                    to_aggr_feat_idx = to_aggr_feat_idx.flatten(1)
                    to_aggr_feat_idx_list[i].append(to_aggr_feat_idx)
        elif self.to_aggr_feat_select_type == 3:
            to_aggr_feat_temp = torch.gather(coarsest_feat, 
                                             dim=0, 
                                             index=to_aggr_feat_idx.permute(1, 0).unsqueeze(-1).expand(-1, -1, self.out_dim))
            to_aggr_feat_list.append(to_aggr_feat_temp.permute(1, 0, 2))
        
        t1 = time.perf_counter()

        for i in range(self.layer_num):

            t2 = time.perf_counter()

            to_re_feat_idx = attn_inuse_sorted_idxs[:, int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[i]): int(attn_inuse_sorted_idxs.size(1) * self.low_score_ratio_list[i])] # [B, num_re]
            curr_attn_threshold = []
            curr_attn_threshold.append(
                torch.gather(
                    attn_inuse,
                    dim=1,
                    index=attn_inuse_sorted_idxs[:, int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[i]):int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[i])+1]))
            curr_attn_threshold.append(
                torch.gather(
                    attn_inuse,
                    dim=1,
                    index=attn_inuse_sorted_idxs[:, int(attn_inuse_sorted_idxs.size(1) * self.low_score_ratio_list[i]):int(attn_inuse_sorted_idxs.size(1) * self.low_score_ratio_list[i])+1]))
            curr_attn_threshold = torch.cat(curr_attn_threshold, dim=1) # [B, 2]
            to_all_feat_coords = pc_coords_list[-(i+2)] # [B, N, 3]
            to_re_feat_coords = torch.gather(to_all_feat_coords,
                                        dim=1,
                                        index=to_re_feat_idx.unsqueeze(-1).expand(-1, -1, 3)) # [B, num_re, 3]
            
            to_re_feat_coords_lazy = LazyTensor(to_re_feat_coords.unsqueeze(-2)) # [B, num_re, 1, 3]
            to_all_feat_coords_lazy = LazyTensor(to_all_feat_coords.unsqueeze(-3)) # [B, 1, N, 3]
            re_to_all_feat_coords_dist_lazy = (to_re_feat_coords_lazy - to_all_feat_coords_lazy).norm2() # [B, num_re, N]
            to_re_feat_idx_inuse = re_to_all_feat_coords_dist_lazy.Kmin_argKmin(
                K=self.same_res_num_neighbor_list[i],
                dim=2,
                device_id=device_id)[1] # [B, num_re, curr_num_neighbor]

            num_re, curr_num_neighbor = to_re_feat_idx_inuse.size(1), to_re_feat_idx_inuse.size(2)
            to_re_feat_attn_inuse = torch.gather(attn_inuse,
                                                dim=1,
                                                index=to_re_feat_idx_inuse.flatten(1)) # [B, num_re * curr_num_neighbor]
            assert torch.count_nonzero(to_re_feat_attn_inuse == 0.) == 0
            to_re_feat_attn_inuse = to_re_feat_attn_inuse.reshape(B, -1, curr_num_neighbor) # [B, num_re, curr_num_neighbor]
            to_re_feat_inuse = torch.gather(coarsest_feat,
                                            dim=0,
                                            index=to_re_feat_idx_inuse.permute(1, 2, 0).flatten(0, 1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [num_re * curr_num_neighbor, B, C]
            to_re_feat_inuse = to_re_feat_inuse.permute(1, 0, 2).reshape(B, num_re, curr_num_neighbor, self.out_dim).reshape(-1, curr_num_neighbor, self.out_dim) # [B * num_re, curr_num_neighbor, C]
            to_re_feat_coords_inuse = torch.gather(to_all_feat_coords,
                                                dim=1,
                                                index=to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, 3)) # [B, num_re * curr_num_neighbor, 3]
            to_re_feat_coords_inuse = to_re_feat_coords_inuse.reshape(B, num_re, curr_num_neighbor, 3) # [B, num_re, curr_num_neighbor, 3]
            next_to_all_feat = pc_feats_list[i+1].permute(0, 2, 1) # [B, C, next_N] -> [B, next_N, C]
            knn_index = index_list[-(i+2)] # [B, N, diff_num_neighbor]
            diff_num_neighbor = knn_index.size(-1)
            next_to_re_feat_idx = torch.gather(knn_index,
                                               dim=1,
                                               index=to_re_feat_idx.unsqueeze(-1).expand(-1, -1, diff_num_neighbor)) # [B, num_re, diff_num_neighbor]
            next_to_all_feat_coords = pc_coords_list[-(i+3)] # [B, next_N, 3]
            next_N = next_to_all_feat.size(1)
            next_to_re_feat_coords = torch.gather(next_to_all_feat_coords,
                                                dim=1,
                                                index=next_to_re_feat_idx.flatten(1).unsqueeze(-1).expand(-1, -1, 3)) # [B, num_re * diff_num_neighbor, 3]
            
            next_to_re_feat_coords_lazy = LazyTensor(next_to_re_feat_coords.unsqueeze(-2)) # [B, num_re * diff_num_neighbor, 1, 3]
            next_to_all_feat_coords_lazy = LazyTensor(next_to_all_feat_coords.unsqueeze(-3)) # [B, 1, next_N, 3]
            next_re_to_next_all_feat_coords_dist_lazy = (next_to_re_feat_coords_lazy - next_to_all_feat_coords_lazy).norm2() # [B, num_re * diff_num_neighbor, next_N]
            next_to_re_feat_idx_inuse = next_re_to_next_all_feat_coords_dist_lazy.Kmin_argKmin(
                K=self.same_res_num_neighbor_list[i+1],
                dim=2,
                device_id=device_id)[1] # [B, num_re * diff_num_neighbor, next_num_neighbor]
            

            next_num_neighbor = next_to_re_feat_idx_inuse.size(-1)
            next_to_re_feat_coords_inuse = torch.gather(next_to_all_feat_coords,
                                                    dim=1,
                                                    index=next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, 3)) # [B, num_re * diff_num_neighbor * next_num_neighbor, 3]
            next_to_re_feat_coords_inuse = next_to_re_feat_coords_inuse.reshape(B, num_re, diff_num_neighbor, next_num_neighbor, 3) # [B, num_re, diff_num_neighbor, next_num_neighbor, 3]
            next_to_re_feat_coords_inuse = next_to_re_feat_coords_inuse.reshape(B * num_re, diff_num_neighbor * next_num_neighbor, 3) # [B * num_re, diff_num_neighbor * next_num_neighbor, 3]

            t3 = time.perf_counter()

            if self.feat_attn_type == 'all_feat':
                pass
                # TODO: to fix
                # next_to_all_feat_num = torch.zeros_like(next_to_all_feat) # [B, next_N, C]
                # next_to_re_feat_num = torch.ones((B, num_re * diff_num_neighbor * next_num_neighbor), dtype=next_to_all_feat_num.dtype, device=device) # [B, num_re * diff_num_neighbor * next_num_neighbor]
                # next_to_all_feat_num.scatter_(dim=1, 
                #                             index=next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim), 
                #                             src=next_to_re_feat_num.unsqueeze(-1).expand(-1, -1, self.out_dim),
                #                             reduce='add') # [B, next_N, C]
                # next_to_all_feat_num = torch.where(next_to_all_feat_num == 0.0, torch.ones_like(next_to_all_feat_num), next_to_all_feat_num)

                # for j in range(self.depth_in_one_layer):
                #     next_to_all_feat = self.SA_blocks_1[i][j](next_to_all_feat, pc_coords_list[-(i+3)])
                #     to_re_feat_inuse = self.SA_blocks_2[i][j](
                #         to_re_feat_inuse, 
                #         to_re_feat_coords_inuse.reshape(-1, curr_num_neighbor, 3))
                #     next_to_re_feat_inuse = torch.gather(next_to_all_feat,
                #                                         dim=1,
                #                                         index=next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [B, num_re * diff_num_neighbor * next_num_neighbor, C]
                #     next_to_re_feat_inuse = next_to_re_feat_inuse.reshape(B, num_re, diff_num_neighbor, next_num_neighbor, self.out_dim) # [B, num_re, diff_num_neighbor, next_num_neighbor, C]
                #     next_to_re_feat_inuse = next_to_re_feat_inuse.reshape(B * num_re, diff_num_neighbor * next_num_neighbor, self.out_dim) # [B * num_re, diff_num_neighbor * next_num_neighbor, C]
                #     (next_to_re_feat_inuse, 
                #     attn1, 
                #     to_re_feat_inuse, 
                #     attn2) = self.CA_blocks[i][j](next_to_re_feat_inuse, 
                #                                 next_to_re_feat_coords_inuse, 
                #                                 to_re_feat_inuse,
                #                                 to_re_feat_coords_inuse.reshape(-1, curr_num_neighbor, 3)) # [B * num_re, diff_num_neighbor * next_num_neighbor, C]、[B * num_re, diff_num_neighbor * next_num_neighbor, curr_num_neighbor]、[B * num_re, curr_num_neighbor, C]、[B * num_re, curr_num_neighbor, diff_num_neighbor * next_num_neighbor]
                #     next_to_all_feat.scatter_(dim=1,          
                #                             index=next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim),  # (B, num_re * diff_num_neighbor * next_num_neighbor, C)
                #                             src=next_to_re_feat_inuse.reshape(B, num_re, diff_num_neighbor, next_num_neighbor, self.out_dim).flatten(1, 3),
                #                             reduce='add') # [B, next_N, C]
                #     next_to_all_feat = next_to_all_feat / next_to_all_feat_num # [B, next_N, C]
            elif self.feat_attn_type == 'only_re_feat':
                next_to_re_feat_inuse = torch.gather(next_to_all_feat,
                                                    dim=1,
                                                    index=next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim)) # [B, num_re * diff_num_neighbor * next_num_neighbor, C]
                next_to_re_feat_inuse = next_to_re_feat_inuse.reshape(B, num_re, diff_num_neighbor, next_num_neighbor, self.out_dim) # [B, num_re, diff_num_neighbor, next_num_neighbor, C]
                next_to_re_feat_inuse = next_to_re_feat_inuse.reshape(B * num_re, diff_num_neighbor * next_num_neighbor, self.out_dim) # [B * num_re, diff_num_neighbor * next_num_neighbor, C]
                for j in range(self.depth_in_one_layer):
                    next_to_re_feat_inuse = self.SA_blocks_1[i][j](next_to_re_feat_inuse, next_to_re_feat_coords_inuse)
                    to_re_feat_inuse = self.SA_blocks_2[i][j](
                        to_re_feat_inuse, 
                        to_re_feat_coords_inuse.reshape(-1, curr_num_neighbor, 3))
                    (next_to_re_feat_inuse, 
                    attn1, 
                    to_re_feat_inuse, 
                    attn2) = self.CA_blocks[i][j](next_to_re_feat_inuse, 
                                                next_to_re_feat_coords_inuse, 
                                                to_re_feat_inuse,
                                                to_re_feat_coords_inuse.reshape(-1, curr_num_neighbor, 3)) # [B * num_re, diff_num_neighbor * next_num_neighbor, C]、[B * num_re, diff_num_neighbor * next_num_neighbor, curr_num_neighbor]、[B * num_re, curr_num_neighbor, C]、[B * num_re, curr_num_neighbor, diff_num_neighbor * next_num_neighbor]
            else:
                raise NotImplementedError

            if self.attn_use_type == 'mean':
                attn = (attn1 + attn2.permute(0, 2, 1)) / 2 # [B * num_re, diff_num_neighbor * next_num_neighbor, curr_num_neighbor]
            elif self.attn_use_type == '1':
                attn = attn1
            elif self.attn_use_type == '2':
                attn = attn2.permute(0, 2, 1)
            else:
                raise NotImplementedError
            attn = attn.reshape(B, num_re, diff_num_neighbor, next_num_neighbor, curr_num_neighbor) # [B, num_re, diff_num_neighbor, next_num_neighbor, curr_num_neighbor]
            positive_mask = torch.gt(to_re_feat_attn_inuse, curr_attn_threshold[:, 0].unsqueeze(-1).unsqueeze(-1)) # [B, num_re, curr_num_neighbor]
            negative_mask = torch.le(to_re_feat_attn_inuse, curr_attn_threshold[:, 1].unsqueeze(-1).unsqueeze(-1)) # [B, num_re, curr_num_neighbor]
            to_re_feat_attn_inuse = torch.where(negative_mask, 1.0 - to_re_feat_attn_inuse, to_re_feat_attn_inuse) # [B, num_re, curr_num_neighbor]
            mask = positive_mask.type(torch.float32) - negative_mask.type(torch.float32)
            to_re_feat_attn_inuse = to_re_feat_attn_inuse * mask # [B, num_re, curr_num_neighbor]
            attn_next = attn * to_re_feat_attn_inuse.unsqueeze(-2).unsqueeze(-2) # [B, num_re, diff_num_neighbor, next_num_neighbor, curr_num_neighbor]
            attn_next = torch.sum(attn_next, dim=-1, keepdim=False) # [B, num_re, diff_num_neighbor, next_num_neighbor]
            attn_next = attn_next.reshape(B, num_re * diff_num_neighbor * next_num_neighbor) # [B, num_re * diff_num_neighbor * next_num_neighbor]
            attn_next_min = torch.min(attn_next, dim=-1, keepdim=False)[0]
            attn_next_min = torch.where(attn_next_min <= 0.0, -attn_next_min, torch.zeros_like(attn_next_min)) + 1e-6
            attn_next = attn_next + attn_next_min.unsqueeze(-1)
            attn_next_max_in_pc = torch.max(attn_next, dim=-1, keepdim=True)[0] # [B, 1]
            attn_next = attn_next / attn_next_max_in_pc # [B, num_re * diff_num_neighbor * next_num_neighbor]
            attn_next = torch.clamp(attn_next, min=1e-6)

            if self.attn_type == 'mean':
                attn_all = torch_scatter.scatter_mean(
                    attn_next, 
                    next_to_re_feat_idx_inuse.flatten(1), 
                    dim=1, 
                    dim_size=next_N,
                    ) # [B, next_N]
            elif self.attn_type == 'max':
                attn_all = torch_scatter.scatter_max(
                    attn_next, 
                    next_to_re_feat_idx_inuse.flatten(1), 
                    dim=1, 
                    dim_size=next_N
                    )[0] # [B, next_N]
            elif self.attn_type == 'min':
                attn_all = torch_scatter.scatter_min(
                    attn_next, 
                    next_to_re_feat_idx_inuse.flatten(1), 
                    dim=1, 
                    dim_size=next_N
                    )[0] # [B, next_N]
            else:
                raise NotImplementedError
            attn_temp = torch.gather(attn_all,
                            dim=-1,
                            index=next_to_re_feat_idx.flatten(1)) # (B, num_re * diff_num_neighbor)
            attn_inuse_sorted_idxs_temp = torch.argsort(attn_temp, dim=-1, descending=True) # [B, num_re * diff_num_neighbor]
            attn_inuse_sorted_idxs = torch.gather(next_to_re_feat_idx.flatten(1),
                                                dim=-1,
                                                index=attn_inuse_sorted_idxs_temp) # [B, num_re * diff_num_neighbor]
            attn_inuse = attn_all
            coarsest_feat = pc_feats_list[i+1].permute(2, 0, 1) # [next_N, B, C]
            to_aggr_feat_idx = attn_inuse_sorted_idxs[:, :int(attn_inuse_sorted_idxs.size(1) * self.high_score_ratio_list[i+1])]

            if self.to_aggr_feat_select_type == 1 or self.to_aggr_feat_select_type == 2:
                to_aggr_feat_idx_list[i+1].append(to_aggr_feat_idx)
                if self.to_aggr_feat_select_type == 2:
                    for j in range(i+2, len(pc_feats_list)):
                        diff_neighbor_num_temp = index_list[-(j+1)].shape[2]
                        to_aggr_feat_idx = torch.gather(index_list[-(j+1)],
                                                        dim=1,
                                                        index=to_aggr_feat_idx.unsqueeze(-1).expand(-1, -1, diff_neighbor_num_temp))
                        to_aggr_feat_idx = to_aggr_feat_idx.flatten(1)
                        to_aggr_feat_idx_list[j].append(to_aggr_feat_idx)
            elif self.to_aggr_feat_select_type == 3:
                next_to_re_feat_inuse_temp = next_to_re_feat_inuse.reshape(B, num_re, diff_num_neighbor, next_num_neighbor, self.out_dim)
                next_to_re_feat_inuse_temp = next_to_re_feat_inuse_temp.reshape(B, num_re * diff_num_neighbor * next_num_neighbor, self.out_dim)
                if self.feat_fuse_type == 'max':
                    next_feats_all_temp = torch_scatter.scatter_max(
                        next_to_re_feat_inuse_temp, 
                        next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim), 
                        dim=1, 
                        dim_size=next_N
                        )[0] # [B, next_N, C]
                elif self.feat_fuse_type == 'min':
                    next_feats_all_temp = torch_scatter.scatter_min(
                        next_to_re_feat_inuse_temp, 
                        next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim), 
                        dim=1, 
                        dim_size=next_N
                        )[0] # [B, next_N, C]
                elif self.feat_fuse_type == 'mean':
                    next_feats_all_temp = torch_scatter.scatter_mean(
                        next_to_re_feat_inuse_temp, 
                        next_to_re_feat_idx_inuse.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim), 
                        dim=1, 
                        dim_size=next_N
                        )
                to_aggr_feat_temp = torch.gather(next_feats_all_temp,
                                                dim=1,
                                                index=next_to_re_feat_idx.flatten(1).unsqueeze(-1).expand(-1, -1, self.out_dim))
                to_aggr_feat_list.append(to_aggr_feat_temp)

            t4 = time.perf_counter()

        t5 = time.perf_counter()

        if self.to_aggr_feat_select_type == 1 or self.to_aggr_feat_select_type == 2:
            return aggr_feat_list, to_aggr_feat_idx_list
        elif self.to_aggr_feat_select_type == 3:
            return aggr_feat_list, to_aggr_feat_list

    
    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)