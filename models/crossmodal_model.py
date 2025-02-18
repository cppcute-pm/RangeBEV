import torch.nn as nn
import torch
from .image_model import ImageEncoder
from .pointcloud_model import PcEncoder
from vision3d.models.geotransformer import SuperPointMatchingMutualTopk, SuperPointProposalGenerator
from vision3d.utils.collate import GraphPyramid2D3DRegistrationCollateFn
from vision3d.array_ops import get_2d3d_correspondences_radius
import torch_scatter
import matplotlib.pyplot as plt
from vision3d.ops import (
    back_project,
    batch_mutual_topk_select,
    create_meshgrid,
    index_select,
    pairwise_cosine_similarity,
    point_to_node_partition,
    point_to_node_partition_batch,
    render,
    pairwise_distance,
)
from .matr2d3d_dp import (
    batchify,
    patchify,
    patchify_CFF,
    get_2d3d_node_correspondences,
    get_2d3d_node_correspondences_batch,
)

from .cmvpr_dp import (
    ResidualAttention, 
    LayerNorm, 
    ResidualSelfAttentionBlock,
    Proj,
    generate_pc_index_and_coords_v1,
    generate_pc_index_and_coords_v2,
    generate_img_index_and_coords_v1,
    generate_img_index_and_knn_and_coords_v3,
    generate_img_meshgrid,
    generate_multi_correspondence_phase4,
    generate_single_correspondence_phase4,
    generate_single_correspondence_phase4_v1,
    generate_single_correspondence_phase4_v2,
    generate_single_correspondence_phase4_v3,
    generate_single_correspondence_in_pair,
    generate_single_correspondence_for_pc,
    generate_cluster_correspondence,
    NMF,
    ImgAttnRe,
    PcAttnRe,
    ResidualSelfAttentionBlock_v2,
    sinusoidal_embedding,
    ClusterTransformer, 
    SemanticTransformer, 
    aggregate_clusterly_and_semantically,
    aggregate_and_match)

from .aggregate_dp import aggregator
import cv2
import torch.nn.functional as F
import random
from .pointnext_dp.layers import create_convblock1d
from copy import deepcopy
import os
import pickle
import copy


class CMVPR(nn.Module):

    def __init__(self, config, out_dim):
        super(CMVPR, self).__init__()
        self.image_encoder = ImageEncoder(config.image_encoder_type, config.image_encoder_cfgs, out_dim, config.image_encoder_out_layer)
        self.pc_encoder = PcEncoder(config.pc_encoder_type, config.pc_encoder_cfgs, out_dim, config.pc_encoder_out_layer)
        self.cfgs = config
        self.pc_aggregator = aggregator(self.cfgs.pc_aggregator_type, self.cfgs.pc_aggregator_cfgs, out_dim)
        self.image_aggregator = aggregator(self.cfgs.image_aggregator_type, self.cfgs.image_aggregator_cfgs, out_dim)
        # self.pcnet.backbone_type = config.pcnet.backbone_type
        # self.pcd_min_node_size = 5
        # self.phase = config.phase # 'single_1'、'single_2'、'cross_1'、'cross_2'、'cross_3'
        # self.matching_radius_2d = 8.0
        if self.cfgs.phase == 2:  # try to create the fusion embedding to enhance the metric learning
            if self.cfgs.phase_fuse == 1: # simply add the features
                pass 
            elif self.cfgs.phase_fuse == 2: # add + mlp
                self.phase2_fuse2_mlp = Proj(embeddim=out_dim,
                                             fuse_type=None,
                                             proj_type=self.cfgs.phase2_proj_type)
            elif self.cfgs.phase_fuse == 3: # concate + mlp
                self.phase2_fuse3_mlp = Proj(embeddim=out_dim,
                                            fuse_type='concat',
                                            proj_type=self.cfgs.phase2_proj_type)
            elif self.cfgs.phase_fuse == 4: # concate + multihead attention: may require the batchsize to be big enough
                self.phase2_fuse4_mha = ResidualSelfAttentionBlock(d_model=2 * out_dim, 
                                                              n_head=self.cfgs.phase2_fuse4_mha_num_heads)
                self.phase2_fuse4_mlp = Proj(embeddim=out_dim,
                                            fuse_type='concat',
                                            proj_type=self.cfgs.phase2_proj_type)
            elif self.cfgs.phase_fuse == 5: # concate + transformer: may require the batchsize to be big enough
                self.phase2_fuse5_transformer = ResidualAttention(num_layers=self.cfgs.phase2_fuse5_trans_num_layers, 
                                                                  d_model=2 * out_dim,
                                                                  n_head=self.cfgs.phase2_fuse5_trans_n_head,
                                                                  att_type='self',
                                                                  out_norm= LayerNorm if self.cfgs.phase2_fuse5_trans_out_norm else None,)
                self.phase2_fuse5_mlp = Proj(embeddim=out_dim,
                                            fuse_type='concat',
                                            proj_type=self.cfgs.phase2_proj_type)
        if self.cfgs.phase == 3:  # try to introduce the projection module and new aggregators, need the pretrained VPR weights
            if self.cfgs.phase_new_PoA == 1: # seperately project local + 2 new aggregators
                self.phase3_new_PoA1_mlp1 = Proj(embeddim=out_dim,
                                                 fuse_type=None,
                                                proj_type=self.cfgs.phase3_proj_type)
                self.phase3_new_PoA1_mlp2 = Proj(embeddim=out_dim,
                                                    fuse_type=None,
                                                    proj_type=self.cfgs.phase3_proj_type)
                self.phase3_new_PoA1_aggregator1 = aggregator(self.cfgs.phase3_aggregator1_type, self.cfgs.phase3_aggregator1_cfgs, out_dim)
                self.phase3_new_PoA1_aggregator2 = aggregator(self.cfgs.phase3_aggregator2_type, self.cfgs.phase3_aggregator2_cfgs, out_dim)
            elif self.cfgs.phase_new_PoA == 2: # only seperately project global
                self.phase3_new_PoA2_mlp1 = Proj(embeddim=out_dim,
                                                 fuse_type=None,
                                                proj_type=self.cfgs.phase3_proj_type)
                self.phase3_new_PoA2_mlp2 = Proj(embeddim=out_dim,
                                                fuse_type=None,
                                                proj_type=self.cfgs.phase3_proj_type)
            elif self.cfgs.phase_new_PoA == 3: # only project global together
                self.phase3_new_PoA3_mlp = Proj(embeddim=out_dim,
                                                fuse_type=None,
                                                proj_type=self.cfgs.phase3_proj_type)
            elif self.cfgs.phase_new_PoA == 4: # 1 new aggregator
                self.phase3_new_PoA4_aggregator = aggregator(self.cfgs.phase3_aggregator_type, self.cfgs.phase3_aggregator_cfgs, out_dim)
            elif self.cfgs.phase_new_PoA == 5: # seperately project local + 1 new aggregator
                self.phase3_new_PoA5_mlp1 = Proj(embeddim=out_dim,
                                                fuse_type=None,
                                                proj_type=self.cfgs.phase3_proj_type)
                self.phase3_new_PoA5_mlp2 = Proj(embeddim=out_dim,
                                                fuse_type=None,
                                                proj_type=self.cfgs.phase3_proj_type)
                self.phase3_new_PoA5_aggregator = aggregator(self.cfgs.phase3_aggregator_type, self.cfgs.phase3_aggregator_cfgs, out_dim)
            elif self.cfgs.phase_new_PoA == 6: # project local together + 1 new aggregator
                self.phase3_new_PoA6_mlp = Proj(embeddim=out_dim,
                                                fuse_type=None,
                                                proj_type=self.cfgs.phase3_proj_type)
                self.phase3_new_PoA6_aggregator = aggregator(self.cfgs.phase3_aggregator_type, self.cfgs.phase3_aggregator_cfgs, out_dim)

        if self.cfgs.phase == 4:
            if 'phase4_multi_aggregator' in self.cfgs.keys() and self.cfgs.phase4_multi_aggregator:
                self.f_pc_aggregator = aggregator(self.cfgs.f_pc_aggregator_type, self.cfgs.f_pc_aggregator_cfgs, out_dim)
                self.f_img_aggregator = aggregator(self.cfgs.f_img_aggregator_type, self.cfgs.f_img_aggregator_cfgs, out_dim)
            if 'phase4_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase4_attention_embeddings:
                self.phase4_attention = ResidualAttention(num_layers=self.cfgs.phase4_attention_num_layers, 
                                                          d_model=out_dim,
                                                          n_head=self.cfgs.phase4_attention_n_head,
                                                          att_type='cross',
                                                          out_norm= LayerNorm if self.cfgs.phase4_attention_out_norm else None,)
        if self.cfgs.phase == 5: 
            self.phase5_aggregator = aggregator(self.cfgs.phase5_aggregator_type, self.cfgs.phase5_aggregator_cfgs, out_dim)
            if 'phase5_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase5_attention_embeddings:
                self.phase5_attention = ResidualAttention(num_layers=self.cfgs.phase5_attention_num_layers, 
                                                            d_model=out_dim,
                                                            n_head=self.cfgs.phase5_attention_n_head,
                                                            att_type='cross',
                                                            out_norm= LayerNorm if self.cfgs.phase5_attention_out_norm else None,)
        
        if self.cfgs.phase == 6:
            if 'phase6_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase6_attention_embeddings:
                self.phase6_attention = ResidualAttention(num_layers=self.cfgs.phase6_attention_num_layers, 
                                                            d_model=out_dim,
                                                            n_head=self.cfgs.phase6_attention_n_head,
                                                            att_type='cross',
                                                            out_norm= LayerNorm if self.cfgs.phase5_attention_out_norm else None,)
        
        if self.cfgs.phase == 7:
            self.phase7_aggregator = aggregator(self.cfgs.phase7_aggregator_type, self.cfgs.phase7_aggregator_cfgs, out_dim)
            if ('phase7_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase7_attention_embeddings) or ('attention_in_local' in self.cfgs.keys() and self.cfgs.attention_in_local == 1):
                self.phase7_attention = ResidualAttention(num_layers=self.cfgs.phase7_attention_num_layers, 
                                                            d_model=out_dim,
                                                            n_head=self.cfgs.phase7_attention_n_head,
                                                            att_type='cross',
                                                            out_norm= LayerNorm if self.cfgs.phase7_attention_out_norm else None,)
        
        if self.cfgs.phase == 8:
            self.phase8_aggregator = aggregator(self.cfgs.phase8_aggregator_type, self.cfgs.phase8_aggregator_cfgs, out_dim)
        
        if self.cfgs.phase == 9:
            if 'phase7_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase7_attention_embeddings:
                self.phase7_attention = ResidualAttention(num_layers=self.cfgs.phase7_attention_num_layers, 
                                                            d_model=out_dim,
                                                            n_head=self.cfgs.phase7_attention_n_head,
                                                            att_type='cross',
                                                            out_norm= LayerNorm if self.cfgs.phase7_attention_out_norm else None,)
        
        if self.cfgs.phase == 11:
            self.phase11_aggregator1 = aggregator(self.cfgs.phase11_aggregator1_type, self.cfgs.phase11_aggregator1_cfgs, out_dim)
            self.phase11_aggregator2 = aggregator(self.cfgs.phase11_aggregator2_type, self.cfgs.phase11_aggregator2_cfgs, out_dim)
            if self.cfgs.phase11_aggregator2_feature_output_type == 'cat' or self.cfgs.phase11_aggregator2_feature_output_type == 'cat_normalize':
                pass
            elif self.cfgs.phase11_aggregator2_feature_output_type == 'fc':
                self.phase11_fc1 = Proj(embeddim=out_dim, 
                                       fuse_type='concat', 
                                       proj_type=self.cfgs.phase11_proj_type1)
                self.phase11_fc2 = Proj(embeddim=out_dim,
                                        fuse_type='concat',
                                        proj_type=self.cfgs.phase11_proj_type2)
            else:
                raise NotImplementedError
        
        if self.cfgs.phase == 12:
            self.phase12_aggregator1 = aggregator(self.cfgs.phase12_aggregator1_type, self.cfgs.phase12_aggregator1_cfgs, out_dim)
            self.phase12_aggregator2 = aggregator(self.cfgs.phase12_aggregator2_type, self.cfgs.phase12_aggregator2_cfgs, out_dim)
            if self.cfgs.phase12_modal_split_aggregator:
                self.phase12_aggregator3 = aggregator(self.cfgs.phase12_aggregator3_type, self.cfgs.phase12_aggregator3_cfgs, out_dim)
                self.phase12_aggregator4 = aggregator(self.cfgs.phase12_aggregator4_type, self.cfgs.phase12_aggregator4_cfgs, out_dim)
        
        if self.cfgs.phase == 13:
            self.phase13_aggregator1 = aggregator(self.cfgs.phase13_aggregator1_type, self.cfgs.phase13_aggregator1_cfgs, out_dim)
            self.phase13_aggregator2 = aggregator(self.cfgs.phase13_aggregator2_type, self.cfgs.phase13_aggregator2_cfgs, out_dim)
        
        if self.cfgs.phase == 14:
            self.phase14_aggregator1 = aggregator(self.cfgs.phase14_aggregator1_type, self.cfgs.phase14_aggregator1_cfgs, out_dim)
            self.phase14_aggregator2 = aggregator(self.cfgs.phase14_aggregator2_type, self.cfgs.phase14_aggregator2_cfgs, out_dim)
            self.phase14_aggregator3 = aggregator(self.cfgs.phase14_aggregator3_type, self.cfgs.phase14_aggregator3_cfgs, out_dim)
            self.phase14_aggregator4 = aggregator(self.cfgs.phase14_aggregator4_type, self.cfgs.phase14_aggregator4_cfgs, out_dim)
            self.phase14_aggregator5 = aggregator(self.cfgs.phase14_aggregator5_type, self.cfgs.phase14_aggregator5_cfgs, out_dim)
        
        if self.cfgs.phase == 15:
            self.phase15_aggregator = aggregator(self.cfgs.phase15_aggregator_type, self.cfgs.phase15_aggregator_cfgs, out_dim)
            self.phase15_aggregator2 = aggregator(self.cfgs.phase15_aggregator2_type, self.cfgs.phase15_aggregator2_cfgs, out_dim)
            if self.cfgs.phase15_attention_type == 1:
                self.phase15_pc_pos_embedding = nn.Sequential(
                                            create_convblock1d(3, 
                                                               128, 
                                                               norm_args=None, 
                                                               act_args={'act': 'gelu'}),
                                            nn.Conv1d(128, 
                                                      out_dim,
                                                      kernel_size=1,)
                                            )
                self.phase15_img_pos_embedding = nn.Sequential(
                                            create_convblock1d(2, 
                                                               128, 
                                                               norm_args=None, 
                                                               act_args={'act': 'gelu'}),
                                            nn.Conv1d(128, 
                                                      out_dim,
                                                      kernel_size=1,)
                                            )
                self.phase15_attention = ResidualAttention(num_layers=self.cfgs.phase15_attention_num_layers, 
                                                            d_model=out_dim,
                                                            n_head=self.cfgs.phase15_attention_n_head,
                                                            att_type='self',
                                                            out_norm=LayerNorm if self.cfgs.phase15_attention_out_norm else None,)
            elif self.cfgs.phase15_attention_type == 2:
                self.phase15_pc_pos_embedding = nn.Sequential(
                                            create_convblock1d(3, 
                                                               128, 
                                                               norm_args=None, 
                                                               act_args={'act': 'gelu'}),
                                            nn.Conv1d(128, 
                                                      out_dim,
                                                      kernel_size=1,)
                                            )
                self.phase15_img_pos_embedding = nn.Sequential(
                                            create_convblock1d(2, 
                                                               128, 
                                                               norm_args=None, 
                                                               act_args={'act': 'gelu'}),
                                            nn.Conv1d(128, 
                                                      out_dim,
                                                      kernel_size=1,)
                                            )
                self.phase15_attention_1 = ResidualAttention(num_layers=self.cfgs.phase15_attention_num_layers, 
                                                            d_model=out_dim,
                                                            n_head=self.cfgs.phase15_attention_n_head,
                                                            att_type='self',
                                                            out_norm= LayerNorm if self.cfgs.phase15_attention_out_norm else None,)
                self.phase15_attention_2 = ResidualAttention(num_layers=self.cfgs.phase15_attention_num_layers, 
                                                            d_model=out_dim,
                                                            n_head=self.cfgs.phase15_attention_n_head,
                                                            att_type='self',
                                                            out_norm= LayerNorm if self.cfgs.phase15_attention_out_norm else None,)
            
            elif self.cfgs.phase15_attention_type == 3:
                self.phase15_pc_pos_embedding = nn.Sequential(
                                            create_convblock1d(3, 
                                                               128, 
                                                               norm_args=None, 
                                                               act_args={'act': 'gelu'}),
                                            nn.Conv1d(128, 
                                                      out_dim,
                                                      kernel_size=1,)
                                            )
                self.phase15_attention = ResidualAttention(num_layers=self.cfgs.phase15_attention_num_layers, 
                                                            d_model=out_dim,
                                                            n_head=self.cfgs.phase15_attention_n_head,
                                                            att_type='self',
                                                            out_norm= LayerNorm if self.cfgs.phase15_attention_out_norm else None,)
            elif self.cfgs.phase15_attention_type == 4:
                self.phase15_attention = ResidualAttention(num_layers=self.cfgs.phase15_attention_num_layers, 
                                                            d_model=out_dim,
                                                            n_head=self.cfgs.phase15_attention_n_head,
                                                            att_type='self',
                                                            out_norm= LayerNorm if self.cfgs.phase15_attention_out_norm else None,)
        
        if self.cfgs.phase == 16:
            self.phase16_aggregator = aggregator(self.cfgs.phase16_aggregator_type, self.cfgs.phase16_aggregator_cfgs, out_dim)
            self.phase16_aggregator_1 = aggregator(self.cfgs.phase16_aggregator1_type, self.cfgs.phase16_aggregator1_cfgs, out_dim)
            self.phase16_aggregator_2 = aggregator(self.cfgs.phase16_aggregator2_type, self.cfgs.phase16_aggregator2_cfgs, out_dim)
            self.phase16_aggregator_3 = aggregator(self.cfgs.phase16_aggregator3_type, self.cfgs.phase16_aggregator3_cfgs, out_dim)
            self.phase16_aggregator_4 = aggregator(self.cfgs.phase16_aggregator4_type, self.cfgs.phase16_aggregator4_cfgs, out_dim)
            self.img_attn_re = ImgAttnRe(self.cfgs.img_attn_re_cfgs, out_dim)
            self.pc_attn_re = PcAttnRe(self.cfgs.pc_attn_re_cfgs, out_dim)
        
        if self.cfgs.phase == 17:
            if self.cfgs.phase17_use_four_aggregator:
                self.phase17_aggregator_1 = aggregator(self.cfgs.phase17_aggregator1_type, self.cfgs.phase17_aggregator1_cfgs, out_dim)
                self.phase17_aggregator_2 = aggregator(self.cfgs.phase17_aggregator2_type, self.cfgs.phase17_aggregator2_cfgs, out_dim)
                self.phase17_aggregator_3 = aggregator(self.cfgs.phase17_aggregator3_type, self.cfgs.phase17_aggregator3_cfgs, out_dim)
                self.phase17_aggregator_4 = aggregator(self.cfgs.phase17_aggregator4_type, self.cfgs.phase17_aggregator4_cfgs, out_dim)
            if self.cfgs.phase17_use_SA_block:
                SA_block = ResidualSelfAttentionBlock_v2(out_dim, self.cfgs.phase17_SA_block_cfgs)
                self.attn_img = nn.ModuleList()
                self.attn_pc = nn.ModuleList()
                for _ in range(self.cfgs.phase17_SA_block_cfgs.num_layers):
                    self.attn_img.append(deepcopy(SA_block))
                    self.attn_pc.append(deepcopy(SA_block))
                self.phase17_img_cls_emb = nn.Parameter(torch.zeros(1, 1, out_dim), requires_grad=True)
                self.phase17_img_pos_emb = nn.Parameter(sinusoidal_embedding(self.cfgs.phase17_img_sequence_length + 1, out_dim), requires_grad=False)
                self.phase17_pc_cls_emb = nn.Parameter(torch.zeros(1, 1, out_dim), requires_grad=True)
                if self.cfgs.phase17_pc_use_pos_emb:
                    self.phase17_pc_pos_emb = nn.Parameter(sinusoidal_embedding(self.cfgs.phase17_pc_sequence_length + 1, out_dim), requires_grad=False)
                if self.cfgs.phase17_SA_block_aggr_type == 1 or self.cfgs.phase17_SA_block_aggr_type == 3 or self.cfgs.phase17_SA_block_aggr_type == 5 or self.cfgs.phase17_SA_block_aggr_type == 6:
                    pass
                elif self.cfgs.phase17_SA_block_aggr_type == 2 or self.cfgs.phase17_SA_block_aggr_type == 4:
                    self.phase17_SA_block_aggregator = aggregator(self.cfgs.phase17_SA_block_aggregator_type, self.cfgs.phase17_SA_block_aggregator_cfgs, out_dim)
                else:
                    raise NotImplementedError
            else:
                self.phase17_aggregator = aggregator(self.cfgs.phase17_aggregator_type, self.cfgs.phase17_aggregator_cfgs, out_dim)
        
        if self.cfgs.phase == 18:
            self.phase18_aggregator = aggregator(self.cfgs.phase18_aggregator_type, self.cfgs.phase18_aggregator_cfgs, out_dim)
            if 'aggregate_clusterly_and_semantically' in self.cfgs.keys():
                self.cluster_transformer = ClusterTransformer(self.cfgs.cluster_transformer_cfgs,
                                                              LayerNorm if self.cfgs.cluster_transformer_cfgs.out_norm else None,)
                self.semantic_transformer = SemanticTransformer(self.cfgs.semantic_transformer_cfgs,
                                                              LayerNorm if self.cfgs.semantic_transformer_cfgs.out_norm else None,)
                if self.cfgs.aggregate_clusterly_and_semantically == 'only_clusterly':
                    self.cluster_aggregator = aggregator(self.cfgs.cluster_aggregator_type, self.cfgs.cluster_aggregator_cfgs, out_dim)
                elif self.cfgs.aggregate_clusterly_and_semantically == 'only_semantically':
                    self.semantic_aggregator = aggregator(self.cfgs.semantic_aggregator_type, self.cfgs.semantic_aggregator_cfgs, out_dim)
                elif self.cfgs.aggregate_clusterly_and_semantically == 'both':
                    self.cluster_aggregator = aggregator(self.cfgs.cluster_aggregator_type, self.cfgs.cluster_aggregator_cfgs, out_dim)
                    self.semantic_aggregator = aggregator(self.cfgs.semantic_aggregator_type, self.cfgs.semantic_aggregator_cfgs, out_dim)
                else:
                    raise NotImplementedError

            if 'phase18_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase18_attention_embeddings:
                pass
        
        if self.cfgs.phase == 19:
            if self.cfgs.phase19_modal_split_aggregator:
                self.phase19_aggregator1_1 = aggregator(self.cfgs.phase19_aggregator1_type, self.cfgs.phase19_aggregator1_cfgs, out_dim)
                self.phase19_aggregator1_2 = aggregator(self.cfgs.phase19_aggregator1_type, self.cfgs.phase19_aggregator1_cfgs, out_dim)
                self.phase19_aggregator2_1 = aggregator(self.cfgs.phase19_aggregator2_type, self.cfgs.phase19_aggregator2_cfgs, out_dim)
                self.phase19_aggregator2_2 = aggregator(self.cfgs.phase19_aggregator2_type, self.cfgs.phase19_aggregator2_cfgs, out_dim)
                self.phase19_aggregator3_1 = aggregator(self.cfgs.phase19_aggregator3_type, self.cfgs.phase19_aggregator3_cfgs, out_dim)
                self.phase19_aggregator3_2 = aggregator(self.cfgs.phase19_aggregator3_type, self.cfgs.phase19_aggregator3_cfgs, out_dim)
                self.phase19_aggregator4_1 = aggregator(self.cfgs.phase19_aggregator4_type, self.cfgs.phase19_aggregator4_cfgs, out_dim)
                self.phase19_aggregator4_2 = aggregator(self.cfgs.phase19_aggregator4_type, self.cfgs.phase19_aggregator4_cfgs, out_dim)
            else:
                self.phase19_aggregator1 = aggregator(self.cfgs.phase19_aggregator1_type, self.cfgs.phase19_aggregator1_cfgs, out_dim)
                self.phase19_aggregator2 = aggregator(self.cfgs.phase19_aggregator2_type, self.cfgs.phase19_aggregator2_cfgs, out_dim)
                self.phase19_aggregator3 = aggregator(self.cfgs.phase19_aggregator3_type, self.cfgs.phase19_aggregator3_cfgs, out_dim)
                self.phase19_aggregator4 = aggregator(self.cfgs.phase19_aggregator4_type, self.cfgs.phase19_aggregator4_cfgs, out_dim)
        
        if self.cfgs.phase == 20:
            SA_block = ResidualSelfAttentionBlock_v2(out_dim, self.cfgs.phase20_SA_block_cfgs)
            self.phase20_attention = nn.ModuleList()
            for _ in range(self.cfgs.phase20_SA_block_cfgs.num_layers):
                self.phase20_attention.append(deepcopy(SA_block))
            self.phase20_img_cls_emb = nn.Parameter(torch.zeros(1, 1, out_dim), requires_grad=True)
            self.phase20_pc_cls_emb = nn.Parameter(torch.zeros(1, 1, out_dim), requires_grad=True)
            if self.cfgs.phase20_SA_block_aggr_type == 1 or self.cfgs.phase20_SA_block_aggr_type == 3 or self.cfgs.phase20_SA_block_aggr_type == 5 or self.cfgs.phase20_SA_block_aggr_type == 6:
                pass
            elif self.cfgs.phase20_SA_block_aggr_type == 2 or self.cfgs.phase20_SA_block_aggr_type == 4:
                self.phase20_SA_block_aggregator = aggregator(self.cfgs.phase20_SA_block_aggregator_type, self.cfgs.phase20_SA_block_aggregator_cfgs, out_dim)
            else:
                raise NotImplementedError
        
        if self.cfgs.phase == 21:
            self.phase21_aggregator = aggregator(self.cfgs.phase21_aggregator_type, self.cfgs.phase21_aggregator_cfgs, out_dim)
        
        if self.cfgs.phase == 22:
            self.phase22_aggregator = aggregator(self.cfgs.phase22_aggregator_type, self.cfgs.phase22_aggregator_cfgs, out_dim)
        
        if self.cfgs.phase == 23:
            self.phase23_aggregator = aggregator(self.cfgs.phase23_aggregator_type, self.cfgs.phase23_aggregator_cfgs, out_dim)
            self.momentum_image_encoder = ImageEncoder(config.image_encoder_type, config.image_encoder_cfgs, out_dim, config.image_encoder_out_layer)
            self.momentum_pc_encoder = PcEncoder(config.pc_encoder_type, config.pc_encoder_cfgs, out_dim, config.pc_encoder_out_layer)
            self.momentum_phase23_aggregator = aggregator(self.cfgs.phase23_aggregator_type, self.cfgs.phase23_aggregator_cfgs, out_dim)
            for param_q, param_k in zip(
                self.pc_encoder.parameters(), self.momentum_pc_encoder.parameters()
            ):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
            for param_q, param_k in zip(
                self.image_encoder.parameters(), self.momentum_image_encoder.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(
                self.phase23_aggregator.parameters(), self.momentum_phase23_aggregator.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
        self.out_dim = out_dim
    
    def forward(self, data_dict):
        """
        data_dict: ['images'、'clouds'、'intrinsics'、'transform']

        """

        # e

        device = data_dict['images'].device
        B, _, img_H, img_W = data_dict["images"].shape
        data_output = {}
        if 'multi_layer' in self.cfgs.keys() and self.cfgs.multi_layer == 3:
            img_1_feats, img_2_feats, img_3_feats = self.image_encoder(data_dict['images'])
            (
                pc_1_feats, 
                pc_2_feats,
                pc_3_feats, 
                points_1, 
                points_2,
                points_3,
                masks_1, 
                masks_2,
                masks_3
            ) = self.pc_encoder(data_dict['clouds'])
        elif 'multi_layer' in self.cfgs.keys() and self.cfgs.multi_layer == 4:
            img_1_feats, img_2_feats, img_3_feats, img_4_feats = self.image_encoder(data_dict['images'])
            (
                pc_1_feats, 
                pc_2_feats,
                pc_3_feats,
                pc_4_feats, 
                points_1, 
                points_2,
                points_3,
                points_4,
            ) = self.pc_encoder(data_dict['clouds'])
        elif 'multi_layer' in self.cfgs.keys() and self.cfgs.multi_layer == 5:
            img_1_feats, img_2_feats, img_3_feats, img_4_feats, img_5_feats = self.image_encoder(data_dict['images'])
            (
                pc_1_feats, 
                pc_2_feats,
                pc_3_feats,
                pc_4_feats, 
                pc_5_feats,
                points_1, 
                points_2,
                points_3,
                points_4,
                points_5,
            ) = self.pc_encoder(data_dict['clouds'])
        elif 'multi_layer' in self.cfgs.keys() and self.cfgs.multi_layer == 6:
            (img_1_feats, 
             img_2_feats, 
             img_2_2_feats, 
             img_3_feats, 
             img_3_2_feats, 
             img_4_feats, 
             img_4_2_feats, 
             img_5_feats) = self.image_encoder(data_dict['images'])
            (
                pc_1_feats, 
                pc_2_feats,
                pc_2_2_feats,
                pc_3_feats,
                pc_3_2_feats,
                pc_4_feats,
                pc_4_2_feats, 
                pc_5_feats,
                points_1, 
                points_2,
                points_3,
                points_4,
                points_5,
            ) = self.pc_encoder(data_dict['clouds'])
        elif 'multi_layer' in self.cfgs.keys() and self.cfgs.multi_layer == 2:
            img_c_feats, img_f_feats1, img_f_feats2 = self.image_encoder(data_dict['images'])
            (
                pc_f_feats1, 
                pc_f_feats2,
                pc_c_feats, 
                f_points, 
                c_points,
            ) = self.pc_encoder(data_dict['clouds'])
        elif 'multi_layer' in self.cfgs.keys() and self.cfgs.multi_layer == 7:
            img_c_feats, img_f_feats1, img_f_feats2 = self.image_encoder(data_dict['images'])
            (
                pc_f_feats1, 
                pc_f_feats2,
                pc_c_feats, 
                points_1, 
                points_2,
                points_3,
                points_4,
            ) = self.pc_encoder(data_dict['clouds'])
        else:
            c_img_feats, f_img_feats = self.image_encoder(data_dict['images'])
            (
                f_pc_feats, 
                c_pc_feats, 
                f_points, 
                c_points, 
                f_masks, 
                c_masks
            ) = self.pc_encoder(data_dict['clouds'])
        if self.cfgs.image_out_layer == 1:
            img_feats_inuse = c_img_feats
        elif self.cfgs.image_out_layer == 2:
            img_feats_inuse = f_img_feats
        elif self.cfgs.image_out_layer == None:
            pass
        if self.cfgs.pc_out_layer == 1:
            pc_feats_inuse = c_pc_feats
            pc_masks_inuse = c_masks
            points_inuse = c_points
        elif self.cfgs.pc_out_layer == 2:
            pc_feats_inuse = f_pc_feats
            pc_masks_inuse = f_masks
            points_inuse = f_points
        elif self.cfgs.pc_out_layer == None:
            pass
        if self.cfgs.phase == 1:
            # use the features directly
            pc_aggr_feat = self.pc_aggregator(pc_feats_inuse, pc_masks_inuse)
            img_aggr_feat = self.image_aggregator(img_feats_inuse)
            data_output['embeddings2'] = pc_aggr_feat
            data_output['embeddings1'] = img_aggr_feat
        elif self.cfgs.phase == 2:
            pc_aggr_feat = self.pc_aggregator(pc_feats_inuse, pc_masks_inuse)
            img_aggr_feat = self.image_aggregator(img_feats_inuse)
            if self.cfgs.phase_fuse == 1:
                fused_feat = pc_aggr_feat + img_aggr_feat
            elif self.cfgs.phase_fuse == 2:
                fused_feat = self.phase2_fuse2_mlp(pc_aggr_feat + img_aggr_feat)
            elif self.cfgs.phase_fuse == 3:
                fused_feat = self.phase2_fuse3_mlp(torch.cat([pc_aggr_feat, img_aggr_feat], dim=1))
            elif self.cfgs.phase_fuse == 4:
                fused_feat = torch.cat([pc_aggr_feat, img_aggr_feat], dim=1)
                fused_feat = self.phase2_fuse4_mha(fused_feat.unsqueeze(1)) # input/output shape: (B, 1, 2 * out_dim)
                fused_feat = self.phase2_fuse4_mlp(fused_feat.squeeze(1))
            elif self.cfgs.phase_fuse == 5:
                fused_feat = torch.cat([pc_aggr_feat, img_aggr_feat], dim=1)
                fused_feat = self.phase2_fuse5_transformer(fused_feat.unsqueeze(0)) # input/output shape: (1, B, 2 * out_dim)
                fused_feat = self.phase2_fuse5_mlp(fused_feat.squeeze(0))
            data_output['embeddings1'] = img_aggr_feat
            data_output['embeddings2'] = pc_aggr_feat
            data_output['embeddings3'] = fused_feat
            data_output['embeddings'] = fused_feat
        elif self.cfgs.phase == 3:
            if self.cfgs.phase_new_PoA == 1:
                c_img_feats_proj = self.phase3_new_PoA1_mlp1(img_feats_inuse)
                c_pc_feats_proj = self.phase3_new_PoA1_mlp2(pc_feats_inuse, pc_masks_inuse)
                img_aggr_feat = self.phase3_new_PoA1_aggregator1(c_img_feats_proj)
                pc_aggr_feat = self.phase3_new_PoA1_aggregator2(c_pc_feats_proj, c_masks)
            elif self.cfgs.phase_new_PoA == 2:
                pc_aggr_feat = self.pc_aggregator(pc_feats_inuse, pc_masks_inuse)
                img_aggr_feat = self.image_aggregator(img_feats_inuse)
                img_aggr_feat = self.phase3_new_PoA2_mlp1(img_aggr_feat)
                pc_aggr_feat = self.phase3_new_PoA2_mlp2(pc_aggr_feat)
            elif self.cfgs.phase_new_PoA == 3:
                pc_aggr_feat = self.pc_aggregator(pc_feats_inuse, pc_masks_inuse)
                img_aggr_feat = self.image_aggregator(img_feats_inuse)
                pc_aggr_feat = self.phase3_new_PoA3_mlp(pc_aggr_feat)
                img_aggr_feat = self.phase3_new_PoA3_mlp(img_aggr_feat)
            elif self.cfgs.phase_new_PoA == 4:
                pc_aggr_feat = self.phase3_new_PoA4_aggregator(pc_feats_inuse, pc_masks_inuse)
                img_aggr_feat = self.phase3_new_PoA4_aggregator(img_feats_inuse)
            elif self.cfgs.phase_new_PoA == 5:
                c_img_feats_proj = self.phase3_new_PoA5_mlp1(img_feats_inuse)
                c_pc_feats_proj = self.phase3_new_PoA5_mlp2(pc_feats_inuse, pc_masks_inuse)
                img_aggr_feat = self.phase3_new_PoA5_aggregator(c_img_feats_proj)
                pc_aggr_feat = self.phase3_new_PoA5_aggregator(c_pc_feats_proj, c_masks)
            elif self.cfgs.phase_new_PoA == 6:
                c_img_feats_proj = self.phase3_new_PoA6_mlp(img_feats_inuse)
                c_pc_feats_proj = self.phase3_new_PoA6_mlp(pc_feats_inuse, pc_masks_inuse)
                img_aggr_feat = self.phase3_new_PoA6_aggregator(c_img_feats_proj)
                pc_aggr_feat = self.phase3_new_PoA6_aggregator(c_pc_feats_proj, c_masks)
            data_output['embeddings1'] = img_aggr_feat
            data_output['embeddings2'] = pc_aggr_feat
        elif self.cfgs.phase == 4:
            if self.training:
                if 'phase4_multi_correspondence' in self.cfgs.keys() and self.cfgs.phase4_multi_correspondence:
                    if 'multi_layer' in self.cfgs.keys() and self.cfgs.multi_layer:
                        img_feats_list = [img_1_feats, img_2_feats, img_3_feats]
                        pc_feats_list = [pc_1_feats, pc_2_feats, pc_3_feats]
                        pc_coords_list = [points_1, points_2, points_3]
                    else:
                        img_feats_list = [f_img_feats, c_img_feats]
                        pc_feats_list = [f_pc_feats, c_pc_feats]
                        pc_coords_list = [f_points, c_points]
                    overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_multi_correspondence_phase4(data_dict, 
                                                        device, 
                                                        img_feats_list, 
                                                        pc_feats_list, 
                                                        pc_coords_list, 
                                                        img_H, 
                                                        img_W, 
                                                        self.cfgs)
                    data_output['img_local_embeddings'] = img_pair_embeddings
                    data_output['pc_local_embeddings'] = pc_pair_embeddings
                    data_output['local_overlap_ratio'] = overlap_ratio_matrix_inuse
                else:
                    overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_single_correspondence_phase4_v2(points_inuse, 
                                                               data_dict, 
                                                               img_H, 
                                                               img_W, 
                                                               device, 
                                                               img_feats_inuse, 
                                                               pc_feats_inuse, 
                                                               self.cfgs)
                    if 'phase4_attention_embeddings' in self.cfgs and self.cfgs.phase4_attention_embeddings:
                        img_pair_embeddings, pc_pair_embeddings = self.phase4_attention(img_pair_embeddings.unsqueeze(0), pc_pair_embeddings.unsqueeze(0))
                        img_pair_embeddings = img_pair_embeddings.squeeze(0)
                        pc_pair_embeddings = pc_pair_embeddings.squeeze(0)
                    data_output['img_local_embeddings1'] = img_pair_embeddings
                    data_output['pc_local_embeddings1'] = pc_pair_embeddings
                    data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
                

            pc_aggr_feat = self.pc_aggregator(pc_feats_inuse, pc_masks_inuse)
            img_aggr_feat = self.image_aggregator(img_feats_inuse)
            data_output['embeddings2'] = pc_aggr_feat
            data_output['embeddings1'] = img_aggr_feat
            if 'phase4_multi_aggregator' in self.cfgs.keys() and self.cfgs.phase4_multi_aggregator:
                if self.cfgs.pc_out_layer == 1:
                    f_pc_aggr_feat = self.f_pc_aggregator(f_pc_feats, f_masks)
                    f_img_aggr_feat = self.f_img_aggregator(f_img_feats)
                    data_output['embeddings3'] = f_img_aggr_feat
                    data_output['embeddings4'] = f_pc_aggr_feat
                elif self.cfgs.pc_out_layer == 2:
                    c_pc_aggr_feat = self.f_pc_aggregator(c_pc_feats, c_masks)
                    c_img_aggr_feat = self.f_img_aggregator(c_img_feats)
                    data_output['embeddings3'] = c_img_aggr_feat
                    data_output['embeddings4'] = c_pc_aggr_feat

            # import numpy as np
            # size = 224
            # block_size = 4
            # checkerboard = np.zeros((size, size), dtype=int)
            # num_blocks = size // block_size
    
            # for i in range(num_blocks):
            #     for j in range(num_blocks):
            #         if (i + j) % 2 == 0:
            #             checkerboard[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 1
            # checkerboard = checkerboard.astype(np.uint8)
            # checkerboard[checkerboard == 1] = 170
            # checkerboard[checkerboard == 0] = 85
            # checkerboard = np.expand_dims(checkerboard, axis=2)

            # img_color = img_2_cimg_idx.reshape(B, B, img_H, img_W) # (B, B, img_H, img_W)
            # pc_pos = data_dict["original_pc_2_many_3"] # (B, B, original_points_num, 2)
            # pc_color = original_pc_2_cpoints # Produces (B, B, original_points_num)
            # pc_mask = torch.logical_and(data_dict["original_pc_2_many_2"][..., 0], data_dict["original_pc_2_many_2"][..., 1]) # Produces (B, B, original_points_num)
            # alpha = 0.4
            # for i in range(B):
            #     for j in range(B):
            #         curr_pc_color = pc_color[i, j, :]
            #         curr_pc_mask = pc_mask[i, j, :]
            #         curr_img_color = img_color[i, j, :, :]
            #         curr_pc_pos = pc_pos[i, j, :, :]
            #         curr_pc_color = np.array(curr_pc_color[curr_pc_mask].to('cpu'))
            #         curr_pc_pos = np.array(curr_pc_pos[curr_pc_mask, :].to('cpu'))
                   
            #         curr_img_color = checkerboard
            #         # curr_img_color = cv2.applyColorMap(checkerboard, cv2.COLORMAP_JET)

            #         fig = plt.figure(figsize=(3.00, 3.00), dpi=100)
            #         ax = fig.add_subplot()
            #         ax.imshow(curr_img_color, alpha=alpha)
            #         ax.set_xlim(0, 224)
            #         ax.set_ylim(224, 0)
            #         ax.scatter(curr_pc_pos[:, 0], curr_pc_pos[:, 1], c=curr_pc_color, marker=',', s=3, edgecolors='none', alpha=0.7, cmap='jet')
            #         text = "555"
            #         position = (50, 50)  # (x, y) 位置
            #         ax.text(position[0], position[1], text, fontsize=12, color='white',)
            #         bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.2')
            #         ax.set_axis_off()
            #         plt.savefig(f'/home/test5/code_project/visualization/boreas_overlap_vis_{i}_{j}.jpg', bbox_inches='tight', pad_inches=0, dpi=200)

            #         # color = clouds_in_camera[i, j, :, 2]
            #         # curr_mask = mask_0[i, j, :] & mask_2[i, j, :]
            #         # color = color[curr_mask]
            #         # # color[~mask_0[i, j, :]] = -500.0
            #         # uv = clouds_in_plane[i, j, :, :]
            #         # uv = uv[curr_mask, :]
            #         # fig = plt.figure(figsize=(30.00, 30.00), dpi=100)
            #         # ax = fig.add_subplot()
            #         # ax.imshow(result['images'][j, :].permute(1, 2, 0))
            #         # # ax.set_xlim(-500, 1000)
            #         # # ax.set_ylim(1000, -500)
            #         # ax.set_xlim(-250, 2500)
            #         # ax.set_ylim(2500, -250)
            #         # ax.scatter(uv[:, 0], uv[:, 1], c=color, marker=',', s=3, edgecolors='none', alpha=0.7, cmap='jet')
            #         # ax.set_axis_off()
            #         # plt.savefig(f'/home/test5/code_project/visualization/heihei_boreas_crop_{i}_{j}.jpg', bbox_inches='tight', pad_inches=0, dpi=200)
        elif self.cfgs.phase == 5:
            pc_aggr_feat = self.phase5_aggregator(pc_feats_inuse, pc_masks_inuse)
            img_aggr_feat = self.phase5_aggregator(img_feats_inuse)
            data_output['embeddings2'] = pc_aggr_feat
            data_output['embeddings1'] = img_aggr_feat
            if self.training:
                if ('phase5_local_correspondence' not in self.cfgs.keys()) or self.cfgs.phase5_local_correspondence:
                    overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_single_correspondence_phase4_v2(points_inuse, 
                                                    data_dict, 
                                                    img_H, 
                                                    img_W, 
                                                    device, 
                                                    img_feats_inuse, 
                                                    pc_feats_inuse, 
                                                    self.cfgs)
                    if 'phase5_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase5_attention_embeddings:
                        img_pair_embeddings, pc_pair_embeddings = self.phase5_attention(img_pair_embeddings.unsqueeze(0), pc_pair_embeddings.unsqueeze(0))
                        img_pair_embeddings = img_pair_embeddings.squeeze(0)
                        pc_pair_embeddings = pc_pair_embeddings.squeeze(0)
                    data_output['img_local_embeddings1'] = img_pair_embeddings
                    data_output['pc_local_embeddings1'] = pc_pair_embeddings
                    data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
        elif self.cfgs.phase == 6:
            if self.training:

                # overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_single_correspondence_phase4(points_inuse, 
                #                                 data_dict, 
                #                                 img_H, 
                #                                 img_W, 
                #                                 device, 
                #                                 img_feats_inuse, 
                #                                 pc_feats_inuse, 
                #                                 self.cfgs)
                # t0 = time.time()
                # overlap_ratio_matrix_inuse_v1, img_pair_embeddings_v1, pc_pair_embeddings_v1 = generate_single_correspondence_phase4_v1(points_inuse, 
                #                                 data_dict, 
                #                                 img_H, 
                #                                 img_W, 
                #                                 device, 
                #                                 img_feats_inuse, 
                #                                 pc_feats_inuse, 
                #                                 self.cfgs)
                # t1 = time.time()
                
                overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_single_correspondence_phase4_v2(points_inuse, 
                                                data_dict, 
                                                img_H, 
                                                img_W, 
                                                device, 
                                                img_feats_inuse, 
                                                pc_feats_inuse, 
                                                self.cfgs)

                if 'phase6_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase6_attention_embeddings:
                    img_pair_embeddings, pc_pair_embeddings = self.phase6_attention(img_pair_embeddings.unsqueeze(0), pc_pair_embeddings.unsqueeze(0))
                    img_pair_embeddings = img_pair_embeddings.squeeze(0)
                    pc_pair_embeddings = pc_pair_embeddings.squeeze(0)
                data_output['img_local_embeddings1'] = img_pair_embeddings
                data_output['pc_local_embeddings1'] = pc_pair_embeddings
                data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
        elif self.cfgs.phase == 7:
            if self.cfgs.phase7_pc_aggr_layer == 1:
                pc_feats_aggr_inuse = c_pc_feats
            elif self.cfgs.phase7_pc_aggr_layer == 2:
                pc_feats_aggr_inuse = f_pc_feats
            elif self.cfgs.phase7_pc_aggr_layer == 3:
                pc_feats_aggr_inuse = pc_c_feats
            elif self.cfgs.phase7_pc_aggr_layer == 4:
                pc_feats_aggr_inuse = pc_f_feats2
            if self.cfgs.phase7_img_aggr_layer == 1:
                img_feats_aggr_inuse = c_img_feats
            elif self.cfgs.phase7_img_aggr_layer == 2:
                img_feats_aggr_inuse = f_img_feats
            elif self.cfgs.phase7_img_aggr_layer == 3:
                img_feats_aggr_inuse = img_c_feats
            elif self.cfgs.phase7_img_aggr_layer == 4:
                img_feats_aggr_inuse = img_f_feats2
            if self.cfgs.phase7_pc_local_layer == 1:
                pc_feats_local_inuse = c_pc_feats
                points_local_inuse = c_points
            elif self.cfgs.phase7_pc_local_layer == 2:
                pc_feats_local_inuse = f_pc_feats
                points_local_inuse = f_points
            elif self.cfgs.phase7_pc_local_layer == 3:
                pc_feats_local_inuse = pc_f_feats1
                points_local_inuse = f_points
            if self.cfgs.phase7_img_local_layer == 1:
                img_feats_local_inuse = c_img_feats
            elif self.cfgs.phase7_img_local_layer == 2:
                img_feats_local_inuse = f_img_feats
            elif self.cfgs.phase7_img_local_layer == 3:
                img_feats_local_inuse = img_f_feats1
                data_output['rgb_depth_preds'] = img_f_feats2.squeeze(1)
            elif self.cfgs.phase7_img_local_layer == 4:
                img_feats_local_inuse = img_f_feats1
            pc_aggr_feat = self.phase7_aggregator(pc_feats_aggr_inuse)
            img_aggr_feat = self.phase7_aggregator(img_feats_aggr_inuse)
            data_output['embeddings2'] = pc_aggr_feat
            data_output['embeddings1'] = img_aggr_feat
            if self.training:
                if ('phase7_local_correspondence' not in self.cfgs.keys()) or self.cfgs.phase7_local_correspondence:
                    if 'attention_in_local' in self.cfgs.keys() and self.cfgs.attention_in_local == 1:
                        (overlap_ratio_matrix_inuse, 
                         img_pair_nonunique_knn_reverse, 
                         img_pair_nonunique_knn_embeddings,
                         img_pair_nonunique_to_unique_idx,
                         pc_pair_nonunique_knn_reverse, 
                         pc_pair_nonunique_knn_embeddings,
                         pc_pair_nonunique_to_unique_idx) = generate_single_correspondence_phase4_v2(points_local_inuse, 
                                                        data_dict, 
                                                        img_H, 
                                                        img_W, 
                                                        device, 
                                                        img_feats_local_inuse, 
                                                        pc_feats_local_inuse, 
                                                        self.cfgs)
                        img_pair_nonunique_knn_embeddings, pc_pair_nonunique_knn_embeddings = self.phase7_attention(img_pair_nonunique_knn_embeddings, 
                                                                                                                    pc_pair_nonunique_knn_embeddings)
                        img_pair_embeddings_nonunique = img_pair_nonunique_knn_embeddings[img_pair_nonunique_knn_reverse[0], img_pair_nonunique_knn_reverse[1]] # (cfgs.phase4_choose_num, out_dim)
                        pc_pair_embeddings_nonunique = pc_pair_nonunique_knn_embeddings[pc_pair_nonunique_knn_reverse[0], pc_pair_nonunique_knn_reverse[1]] # (cfgs.phase4_choose_num, out_dim)
                        img_pair_embeddings = torch_scatter.scatter_mean(img_pair_embeddings_nonunique, 
                                                                        img_pair_nonunique_to_unique_idx.unsqueeze(1).expand(-1, self.out_dim), 
                                                                        dim=0) # (num_img_1, out_dim)
                        pc_pair_embeddings = torch_scatter.scatter_mean(pc_pair_embeddings_nonunique, 
                                                pc_pair_nonunique_to_unique_idx.unsqueeze(1).expand(-1, self.out_dim), 
                                                dim=0) # (num_pc_1, out_dim)
                        data_output['img_local_embeddings1'] = img_pair_embeddings
                        data_output['pc_local_embeddings1'] = pc_pair_embeddings
                        data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
                    elif 'attention_in_local' in self.cfgs.keys() and self.cfgs.attention_in_local == 2:
                        (overlap_ratio_matrix_inuse, 
                         img_pair_nonunique_knn_embeddings,
                         pc_pair_nonunique_knn_embeddings) = generate_single_correspondence_phase4_v2(points_local_inuse, 
                                                        data_dict, 
                                                        img_H, 
                                                        img_W, 
                                                        device, 
                                                        img_feats_local_inuse, 
                                                        pc_feats_local_inuse, 
                                                        self.cfgs)
                        if 'phase7_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase7_attention_embeddings:
                            (img_pair_nonunique_knn_embeddings, 
                             pc_pair_nonunique_knn_embeddings) = self.phase7_attention(img_pair_nonunique_knn_embeddings, 
                                                                                        pc_pair_nonunique_knn_embeddings)
                        
                        max_in_img_local = torch.max(overlap_ratio_matrix_inuse, dim=2)[0] # (cfgs.phase4_choose_num, pc_attention_in_local_k)
                        max_in_all_local = torch.max(max_in_img_local, dim=1)[0] # (cfgs.phase4_choose_num, )
                        overlap_ratio_matrix_inuse = overlap_ratio_matrix_inuse / (max_in_all_local.unsqueeze(1).unsqueeze(2) + 1e-6)
                        data_output['img_local_embeddings1'] = img_pair_nonunique_knn_embeddings
                        data_output['pc_local_embeddings1'] = pc_pair_nonunique_knn_embeddings
                        data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
                    else:
                        overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_single_correspondence_phase4_v2(points_local_inuse, 
                                                        data_dict, 
                                                        img_H, 
                                                        img_W, 
                                                        device, 
                                                        img_feats_local_inuse, 
                                                        pc_feats_local_inuse, 
                                                        self.cfgs)
                        if 'phase7_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase7_attention_embeddings:
                            img_pair_embeddings, pc_pair_embeddings = self.phase7_attention(img_pair_embeddings.unsqueeze(0), pc_pair_embeddings.unsqueeze(0))
                            img_pair_embeddings = img_pair_embeddings.squeeze(0)
                            pc_pair_embeddings = pc_pair_embeddings.squeeze(0)
                        data_output['img_local_embeddings1'] = img_pair_embeddings
                        data_output['pc_local_embeddings1'] = pc_pair_embeddings
                        data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
        elif self.cfgs.phase == 8:
            pc_feats_aggr_inuse = pc_1_feats
            img_feats_aggr_inuse = img_1_feats
            if 'phase8_aggr_types' in self.cfgs.keys() and self.cfgs.phase8_aggr_types:
                pc_aggr_feat = self.pc_aggregator(pc_feats_aggr_inuse)
                img_aggr_feat = self.image_aggregator(img_feats_aggr_inuse)
            else:
                pc_aggr_feat = self.phase8_aggregator(pc_feats_aggr_inuse)
                img_aggr_feat = self.phase8_aggregator(img_feats_aggr_inuse)
            data_output['embeddings2'] = pc_aggr_feat
            data_output['embeddings1'] = img_aggr_feat
            if self.training:
                pc_feats_list = [pc_2_feats, pc_3_feats, pc_4_feats]
                pc_points_list = [points_2, points_3, points_4]
                img_feats_list = [img_2_feats, img_3_feats, img_4_feats]
                overlap_ratio_matrix_inuse_list = []
                img_pair_embeddings_list = []
                pc_pair_embeddings_list = []
                for i in range(3):
                    curr_points = pc_points_list[i]
                    curr_img_feats = img_feats_list[i]
                    curr_pc_feats = pc_feats_list[i]
                    curr_min_pc_num_pt = self.cfgs.phase8_min_pc_num_pt[i]
                    curr_min_img_num_pt = self.cfgs.phase8_min_img_num_pt[i]
                    curr_topk = self.cfgs.phase8_topk[i]
                    curr_overlap_ratio_matrix, curr_img_pair_embeddings, curr_pc_pair_embeddings = generate_single_correspondence_phase4_v3(curr_points, 
                                                    data_dict, 
                                                    img_H, 
                                                    img_W, 
                                                    device, 
                                                    curr_img_feats,
                                                    curr_pc_feats, 
                                                    self.cfgs,
                                                    curr_min_pc_num_pt,
                                                    curr_min_img_num_pt,
                                                    curr_topk)
                    overlap_ratio_matrix_inuse_list.append(curr_overlap_ratio_matrix)
                    img_pair_embeddings_list.append(curr_img_pair_embeddings)
                    pc_pair_embeddings_list.append(curr_pc_pair_embeddings)
                data_output['img_local_embeddings'] = img_pair_embeddings_list
                data_output['pc_local_embeddings'] = pc_pair_embeddings_list
                data_output['local_overlap_ratio'] = overlap_ratio_matrix_inuse_list
        elif self.cfgs.phase == 9:
            if self.cfgs.phase7_pc_aggr_layer == 1:
                pc_feats_aggr_inuse = c_pc_feats
                pc_masks_aggr_inuse = c_masks
                points_aggr_inuse = c_points
            elif self.cfgs.phase7_pc_aggr_layer == 2:
                pc_feats_aggr_inuse = f_pc_feats
                pc_masks_aggr_inuse = f_masks
                points_aggr_inuse = f_points
            if self.cfgs.phase7_img_aggr_layer == 1:
                img_feats_aggr_inuse = c_img_feats
            elif self.cfgs.phase7_img_aggr_layer == 2:
                img_feats_aggr_inuse = f_img_feats
            if self.cfgs.phase7_pc_local_layer == 1:
                pc_feats_local_inuse = c_pc_feats
                pc_masks_local_inuse = c_masks
                points_local_inuse = c_points
            elif self.cfgs.phase7_pc_local_layer == 2:
                pc_feats_local_inuse = f_pc_feats
                pc_masks_local_inuse = f_masks
                points_local_inuse = f_points
            if self.cfgs.phase7_img_local_layer == 1:
                img_feats_local_inuse = c_img_feats
            elif self.cfgs.phase7_img_local_layer == 2:
                img_feats_local_inuse = f_img_feats
            pc_aggr_feat = self.pc_aggregator(pc_feats_aggr_inuse, pc_masks_aggr_inuse)
            img_aggr_feat = self.image_aggregator(img_feats_aggr_inuse)
            data_output['embeddings2'] = pc_aggr_feat
            data_output['embeddings1'] = img_aggr_feat
            if self.training:
                if ('phase7_local_correspondence' not in self.cfgs.keys()) or self.cfgs.phase7_local_correspondence:
                    overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_single_correspondence_phase4_v2(points_local_inuse, 
                                                    data_dict, 
                                                    img_H, 
                                                    img_W, 
                                                    device, 
                                                    img_feats_local_inuse, 
                                                    pc_feats_local_inuse, 
                                                    self.cfgs)
                    if 'phase7_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase7_attention_embeddings:
                        img_pair_embeddings, pc_pair_embeddings = self.phase7_attention(img_pair_embeddings.unsqueeze(0), pc_pair_embeddings.unsqueeze(0))
                        img_pair_embeddings = img_pair_embeddings.squeeze(0)
                        pc_pair_embeddings = pc_pair_embeddings.squeeze(0)
                    data_output['img_local_embeddings1'] = img_pair_embeddings
                    data_output['pc_local_embeddings1'] = pc_pair_embeddings
                    data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
        elif self.cfgs.phase == 11:
            pc_aggr_feat_1 = self.phase11_aggregator1(pc_c_feats)
            img_aggr_feat_1 = self.phase11_aggregator1(img_c_feats)
            pc_aggr_feat_2 = self.phase11_aggregator2(pc_f_feats2)
            img_aggr_feat_2 = self.phase11_aggregator2(img_f_feats2)
            if self.cfgs.phase11_aggregator2_feature_output_type == 'cat':
                img_aggr_feat = torch.cat([img_aggr_feat_1, img_aggr_feat_2], dim=-1)
                pc_aggr_feat = torch.cat([pc_aggr_feat_1, pc_aggr_feat_2], dim=-1)
            elif self.cfgs.phase11_aggregator2_feature_output_type == 'fc':
                img_aggr_feat = self.phase11_fc1(torch.cat([img_aggr_feat_1, img_aggr_feat_2], dim=-1))
                pc_aggr_feat = self.phase11_fc2(torch.cat([pc_aggr_feat_1, pc_aggr_feat_2], dim=-1))
            elif self.cfgs.phase11_aggregator2_feature_output_type == 'cat_normalize':
                img_aggr_feat_1 = F.normalize(img_aggr_feat_1, p=2, dim=-1)
                img_aggr_feat_2 = F.normalize(img_aggr_feat_2, p=2, dim=-1)
                pc_aggr_feat_1 = F.normalize(pc_aggr_feat_1, p=2, dim=-1)
                pc_aggr_feat_2 = F.normalize(pc_aggr_feat_2, p=2, dim=-1)
                img_aggr_feat = torch.cat([img_aggr_feat_1, img_aggr_feat_2], dim=-1)
                pc_aggr_feat = torch.cat([pc_aggr_feat_1, pc_aggr_feat_2], dim=-1)
            data_output['embeddings2'] = pc_aggr_feat
            data_output['embeddings1'] = img_aggr_feat
            if self.training:
                if ('phase11_local_correspondence' not in self.cfgs.keys()) or self.cfgs.phase11_local_correspondence:
                    overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_single_correspondence_phase4_v2(f_points, 
                                                    data_dict, 
                                                    img_H, 
                                                    img_W, 
                                                    device, 
                                                    img_f_feats1, 
                                                    pc_f_feats1, 
                                                    self.cfgs)
                    data_output['img_local_embeddings1'] = img_pair_embeddings
                    data_output['pc_local_embeddings1'] = pc_pair_embeddings
                    data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
        elif self.cfgs.phase == 12:
            if 'phase12_pos_aggregator1' in self.cfgs.keys() and self.cfgs.phase12_pos_aggregator1:
                if 'phase12_feats_inuse' in self.cfgs.keys() and self.cfgs.phase12_feats_inuse == 1:
                    coords_list = [points_1]
                    pc_index_list_1, pc_coords_list_1 = generate_pc_index_and_coords_v2(coords_list)
                else:
                    pc_index_list_1, pc_coords_list_1 = generate_pc_index_and_coords_v1(self.cfgs.phase12_coords_num_list_1, c_points)
                img_index_list_1, img_coords_list_1 = generate_img_index_and_coords_v1(B, self.cfgs.phase12_resolution_list_1, device)
                if self.cfgs.phase12_modal_split_aggregator:
                    pc_aggr_feat_1 = self.phase12_aggregator1(pc_c_feats, mask=None, index_list=pc_index_list_1, coords_list=pc_coords_list_1)
                    img_c_feats = torch.flatten(img_c_feats, start_dim=2)
                    img_aggr_feat_1 = self.phase12_aggregator2(img_c_feats, mask=None, index_list=img_index_list_1, coords_list=img_coords_list_1)
                else:
                    pc_aggr_feat_1 = self.phase12_aggregator1(pc_c_feats, mask=None, index_list=pc_index_list_1, coords_list=pc_coords_list_1)
                    img_c_feats = torch.flatten(img_c_feats, start_dim=2)
                    img_aggr_feat_1 = self.phase12_aggregator1(img_c_feats, mask=None, index_list=img_index_list_1, coords_list=img_coords_list_1)
            else:
                pc_aggr_feat_1 = self.phase12_aggregator1(pc_c_feats)
                img_aggr_feat_1 = self.phase12_aggregator1(img_c_feats)
            if 'phase12_pos_aggregator2' in self.cfgs.keys() and self.cfgs.phase12_pos_aggregator2:
                if 'phase12_feats_inuse' in self.cfgs.keys() and self.cfgs.phase12_feats_inuse == 1:
                    coords_list = [points_3, points_2, points_1]
                    pc_index_list_2, pc_coords_list_2 = generate_pc_index_and_coords_v2(coords_list)
                    f_points = points_3
                else:
                    pc_index_list_2, pc_coords_list_2 = generate_pc_index_and_coords_v1(self.cfgs.phase12_coords_num_list_2, f_points)
                img_index_list_2, img_coords_list_2 = generate_img_index_and_coords_v1(B, self.cfgs.phase12_resolution_list_2, device)
                if self.cfgs.phase12_modal_split_aggregator:
                    pc_aggr_feat_2 = self.phase12_aggregator3(pc_f_feats2, mask=None, index_list=pc_index_list_2, coords_list=pc_coords_list_2)
                    img_f_feats2 = torch.flatten(img_f_feats2, start_dim=2)
                    img_aggr_feat_2 = self.phase12_aggregator4(img_f_feats2, mask=None, index_list=img_index_list_2, coords_list=img_coords_list_2)
                else:
                    pc_aggr_feat_2 = self.phase12_aggregator2(pc_f_feats2, mask=None, index_list=pc_index_list_2, coords_list=pc_coords_list_2)
                    img_f_feats2 = torch.flatten(img_f_feats2, start_dim=2)
                    img_aggr_feat_2 = self.phase12_aggregator2(img_f_feats2, mask=None, index_list=img_index_list_2, coords_list=img_coords_list_2)
            else:
                pc_aggr_feat_2 = self.phase12_aggregator2(pc_f_feats2)
                img_aggr_feat_2 = self.phase12_aggregator2(img_f_feats2)
            if self.training and self.cfgs.phase12_feature_train_type == 'split':
                data_output['embeddings2'] = pc_aggr_feat_1
                data_output['embeddings1'] = img_aggr_feat_1
                data_output['embeddings4'] = pc_aggr_feat_2
                data_output['embeddings3'] = img_aggr_feat_2
            elif self.cfgs.phase12_feature_train_type == 'concat':
                data_output['embeddings2'] = torch.cat([pc_aggr_feat_1, pc_aggr_feat_2], dim=-1)
                data_output['embeddings1'] = torch.cat([img_aggr_feat_1, img_aggr_feat_2], dim=-1)
            else:
                data_output['embeddings2'] = pc_aggr_feat_2
                data_output['embeddings1'] = img_aggr_feat_2
            if self.training:
                if ('phase12_local_correspondence' not in self.cfgs.keys()) or self.cfgs.phase12_local_correspondence:
                    overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_single_correspondence_phase4_v2(f_points, 
                                                    data_dict, 
                                                    img_H, 
                                                    img_W, 
                                                    device, 
                                                    img_f_feats1, 
                                                    pc_f_feats1, 
                                                    self.cfgs)
                    data_output['img_local_embeddings1'] = img_pair_embeddings
                    data_output['pc_local_embeddings1'] = pc_pair_embeddings
                    data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
        elif self.cfgs.phase == 13:
            pc_aggr_feat_1 = self.phase13_aggregator1(pc_1_feats)
            img_aggr_feat_1 = self.phase13_aggregator1(img_1_feats)
            pc_aggr_feat_2 = self.phase13_aggregator2(pc_5_feats)
            img_aggr_feat_2 = self.phase13_aggregator2(img_5_feats)
            if self.training and self.cfgs.phase13_feature_train_type == 'split':
                data_output['embeddings2'] = pc_aggr_feat_1
                data_output['embeddings1'] = img_aggr_feat_1
                data_output['embeddings4'] = pc_aggr_feat_2
                data_output['embeddings3'] = img_aggr_feat_2
            elif self.cfgs.phase13_feature_train_type == 'concat':
                data_output['embeddings2'] = torch.cat([pc_aggr_feat_1, pc_aggr_feat_2], dim=-1)
                data_output['embeddings1'] = torch.cat([img_aggr_feat_1, img_aggr_feat_2], dim=-1)
            else:
                data_output['embeddings2'] = pc_aggr_feat_2
                data_output['embeddings1'] = img_aggr_feat_2
            if self.training:
                if 'phase13_local_correspondence_one_layer' in self.cfgs.keys() and self.cfgs.phase13_local_correspondence_one_layer:
                    overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_single_correspondence_phase4_v3(points_4, 
                                                    data_dict, 
                                                    img_H, 
                                                    img_W, 
                                                    device, 
                                                    img_4_feats, 
                                                    pc_4_feats, 
                                                    self.cfgs,
                                                    self.cfgs.phase13_min_pc_num_pt[2],
                                                    self.cfgs.phase13_min_img_num_pt[2],
                                                    self.cfgs.phase13_topk[2])
                    data_output['img_local_embeddings1'] = img_pair_embeddings
                    data_output['pc_local_embeddings1'] = pc_pair_embeddings
                    data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
                else:
                    pc_feats_list = [pc_2_feats, pc_3_feats, pc_4_feats]
                    pc_points_list = [points_2, points_3, points_4]
                    img_feats_list = [img_2_feats, img_3_feats, img_4_feats]
                    overlap_ratio_matrix_inuse_list = []
                    img_pair_embeddings_list = []
                    pc_pair_embeddings_list = []
                    for i in range(3):
                        curr_points = pc_points_list[i]
                        curr_img_feats = img_feats_list[i]
                        curr_pc_feats = pc_feats_list[i]
                        curr_min_pc_num_pt = self.cfgs.phase13_min_pc_num_pt[i]
                        curr_min_img_num_pt = self.cfgs.phase13_min_img_num_pt[i]
                        curr_topk = self.cfgs.phase13_topk[i]
                        curr_overlap_ratio_matrix, curr_img_pair_embeddings, curr_pc_pair_embeddings = generate_single_correspondence_phase4_v3(curr_points, 
                                                        data_dict, 
                                                        img_H, 
                                                        img_W, 
                                                        device, 
                                                        curr_img_feats,
                                                        curr_pc_feats, 
                                                        self.cfgs,
                                                        curr_min_pc_num_pt,
                                                        curr_min_img_num_pt,
                                                        curr_topk)
                        overlap_ratio_matrix_inuse_list.append(curr_overlap_ratio_matrix)
                        img_pair_embeddings_list.append(curr_img_pair_embeddings)
                        pc_pair_embeddings_list.append(curr_pc_pair_embeddings)
                    data_output['img_local_embeddings'] = img_pair_embeddings_list
                    data_output['pc_local_embeddings'] = pc_pair_embeddings_list
                    data_output['local_overlap_ratio'] = overlap_ratio_matrix_inuse_list
        elif self.cfgs.phase == 14:
            pc_aggr_feat_1 = self.phase14_aggregator1(pc_1_feats)
            img_aggr_feat_1 = self.phase14_aggregator1(img_1_feats)
            pc_aggr_feat_2 = self.phase14_aggregator2(pc_2_2_feats)
            img_aggr_feat_2 = self.phase14_aggregator2(img_2_2_feats)
            pc_aggr_feat_3 = self.phase14_aggregator3(pc_3_2_feats)
            img_aggr_feat_3 = self.phase14_aggregator3(img_3_2_feats)
            pc_aggr_feat_4 = self.phase14_aggregator4(pc_4_2_feats)
            img_aggr_feat_4 = self.phase14_aggregator4(img_4_2_feats)
            pc_aggr_feat_5 = self.phase14_aggregator5(pc_5_feats)
            img_aggr_feat_5 = self.phase14_aggregator5(img_5_feats)
            if self.training and self.cfgs.phase14_feature_train_type == 'split':
                data_output['embeddings2'] = pc_aggr_feat_1
                data_output['embeddings1'] = img_aggr_feat_1
                data_output['embeddings4'] = pc_aggr_feat_2
                data_output['embeddings3'] = img_aggr_feat_2
                data_output['embeddings6'] = pc_aggr_feat_3
                data_output['embeddings5'] = img_aggr_feat_3
                data_output['embeddings8'] = pc_aggr_feat_4
                data_output['embeddings7'] = img_aggr_feat_4
                data_output['embeddings10'] = pc_aggr_feat_5
                data_output['embeddings9'] = img_aggr_feat_5
            elif self.cfgs.phase14_feature_train_type == 'concat':
                data_output['embeddings2'] = torch.cat([pc_aggr_feat_1, pc_aggr_feat_2, pc_aggr_feat_3, pc_aggr_feat_4, pc_aggr_feat_5], dim=-1)
                data_output['embeddings1'] = torch.cat([img_aggr_feat_1, img_aggr_feat_2, img_aggr_feat_3, img_aggr_feat_4, img_aggr_feat_5], dim=-1)
            else:
                data_output['embeddings2'] = pc_aggr_feat_5
                data_output['embeddings1'] = img_aggr_feat_5
            if self.training:
                if 'phase14_local_correspondence_one_layer' in self.cfgs.keys() and self.cfgs.phase14_local_correspondence_one_layer:
                    overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_single_correspondence_phase4_v3(points_4, 
                                                    data_dict, 
                                                    img_H, 
                                                    img_W, 
                                                    device, 
                                                    img_4_feats, 
                                                    pc_4_feats, 
                                                    self.cfgs,
                                                    self.cfgs.phase14_min_pc_num_pt[2],
                                                    self.cfgs.phase14_min_img_num_pt[2],
                                                    self.cfgs.phase14_topk[2])
                    data_output['img_local_embeddings1'] = img_pair_embeddings
                    data_output['pc_local_embeddings1'] = pc_pair_embeddings
                    data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
                else:
                    pc_feats_list = [pc_2_feats, pc_3_feats, pc_4_feats]
                    pc_points_list = [points_2, points_3, points_4]
                    img_feats_list = [img_2_feats, img_3_feats, img_4_feats]
                    overlap_ratio_matrix_inuse_list = []
                    img_pair_embeddings_list = []
                    pc_pair_embeddings_list = []
                    for i in range(3):
                        curr_points = pc_points_list[i]
                        curr_img_feats = img_feats_list[i]
                        curr_pc_feats = pc_feats_list[i]
                        curr_min_pc_num_pt = self.cfgs.phase14_min_pc_num_pt[i]
                        curr_min_img_num_pt = self.cfgs.phase14_min_img_num_pt[i]
                        curr_topk = self.cfgs.phase14_topk[i]
                        curr_overlap_ratio_matrix, curr_img_pair_embeddings, curr_pc_pair_embeddings = generate_single_correspondence_phase4_v3(curr_points, 
                                                        data_dict, 
                                                        img_H, 
                                                        img_W, 
                                                        device, 
                                                        curr_img_feats,
                                                        curr_pc_feats, 
                                                        self.cfgs,
                                                        curr_min_pc_num_pt,
                                                        curr_min_img_num_pt,
                                                        curr_topk)
                        overlap_ratio_matrix_inuse_list.append(curr_overlap_ratio_matrix)
                        img_pair_embeddings_list.append(curr_img_pair_embeddings)
                        pc_pair_embeddings_list.append(curr_pc_pair_embeddings)
                    data_output['img_local_embeddings'] = img_pair_embeddings_list
                    data_output['pc_local_embeddings'] = pc_pair_embeddings_list
                    data_output['local_overlap_ratio'] = overlap_ratio_matrix_inuse_list
        
        elif self.cfgs.phase == 15:
            if self.cfgs.phase15_pc_aggr_layer == 1:
                pc_feats_aggr_inuse = c_pc_feats
            elif self.cfgs.phase15_pc_aggr_layer == 2:
                pc_feats_aggr_inuse = f_pc_feats
            elif self.cfgs.phase15_pc_aggr_layer == 3:
                pc_feats_aggr_inuse = pc_c_feats
            elif self.cfgs.phase15_pc_aggr_layer == 4:
                pc_feats_aggr_inuse = pc_f_feats2
            if self.cfgs.phase15_img_aggr_layer == 1:
                img_feats_aggr_inuse = c_img_feats
            elif self.cfgs.phase15_img_aggr_layer == 2:
                img_feats_aggr_inuse = f_img_feats
            elif self.cfgs.phase15_img_aggr_layer == 3:
                img_feats_aggr_inuse = img_c_feats
            elif self.cfgs.phase15_img_aggr_layer == 4:
                img_feats_aggr_inuse = img_f_feats2
            if self.cfgs.phase15_pc_local_layer == 1:
                pc_feats_local_inuse = c_pc_feats
                points_local_inuse = c_points
            elif self.cfgs.phase15_pc_local_layer == 2:
                pc_feats_local_inuse = f_pc_feats
                points_local_inuse = f_points
            elif self.cfgs.phase15_pc_local_layer == 3:
                pc_feats_local_inuse = pc_f_feats1
                points_local_inuse = f_points
            if self.cfgs.phase15_img_local_layer == 1:
                img_feats_local_inuse = c_img_feats
            elif self.cfgs.phase15_img_local_layer == 2:
                img_feats_local_inuse = f_img_feats
            elif self.cfgs.phase15_img_local_layer == 3:
                img_feats_local_inuse = img_f_feats1
                data_output['rgb_depth_preds'] = img_f_feats2.squeeze(1)
            elif self.cfgs.phase15_img_local_layer == 4:
                img_feats_local_inuse = img_f_feats1
            pc_aggr_feat = self.phase15_aggregator(pc_feats_aggr_inuse)
            img_aggr_feat = self.phase15_aggregator(img_feats_aggr_inuse)
            if self.cfgs.phase15_attention_type == 1:
                pc_pos_embeddings = self.phase15_pc_pos_embedding(f_points.permute(0, 2, 1)) # (B, out_dim, N)
                pc_f_feats2 = pc_f_feats2 + pc_pos_embeddings
                f_img_size = img_f_feats2.shape[2:4]
                original_img_size = [img_H, img_W]
                f_meshgrid = generate_img_meshgrid(B, f_img_size, original_img_size, device)
                img_pos_embeddings = self.phase15_img_pos_embedding(f_meshgrid.permute(0, 2, 1)) # (B, out_dim, curr_img_H * curr_img_W)
                img_f_feats2 = img_f_feats2.reshape(B, -1, f_img_size[0] * f_img_size[1])
                img_f_feats2 = img_f_feats2 + img_pos_embeddings
                pc_f_feats2 = self.phase15_attention(x=pc_f_feats2.permute(0, 2, 1))
                img_f_feats2 = self.phase15_attention(x=img_f_feats2.permute(0, 2, 1))
                pc_aggr_feat_2 = self.phase15_aggregator2(pc_f_feats2.permute(0, 2, 1))
                img_aggr_feat_2 = self.phase15_aggregator2(img_f_feats2.permute(0, 2, 1))
            elif self.cfgs.phase15_attention_type == 2:
                pc_pos_embeddings = self.phase15_pc_pos_embedding(f_points.permute(0, 2, 1)) # (B, out_dim, N)
                pc_f_feats2 = pc_f_feats2 + pc_pos_embeddings
                f_img_size = img_f_feats2.shape[2:4]
                original_img_size = [img_H, img_W]
                f_meshgrid = generate_img_meshgrid(B, f_img_size, original_img_size, device)
                img_pos_embeddings = self.phase15_img_pos_embedding(f_meshgrid.permute(0, 2, 1)) # (B, out_dim, curr_img_H * curr_img_W)
                img_f_feats2 = img_f_feats2.reshape(B, -1, f_img_size[0] * f_img_size[1])
                img_f_feats2 = img_f_feats2 + img_pos_embeddings
                pc_f_feats2 = self.phase15_attention_1(x=pc_f_feats2.permute(0, 2, 1))
                img_f_feats2 = self.phase15_attention_2(x=img_f_feats2.permute(0, 2, 1))
                pc_aggr_feat_2 = self.phase15_aggregator2(pc_f_feats2.permute(0, 2, 1))
                img_aggr_feat_2 = self.phase15_aggregator2(img_f_feats2.permute(0, 2, 1))
            elif self.cfgs.phase15_attention_type == 3:
                pc_pos_embeddings = self.phase15_pc_pos_embedding(f_points.permute(0, 2, 1)) # (B, out_dim, N)
                pc_f_feats2 = pc_f_feats2 + pc_pos_embeddings
                f_img_size = img_f_feats2.shape[2:4]
                img_f_feats2 = img_f_feats2.reshape(B, -1, f_img_size[0] * f_img_size[1])
                pc_f_feats2 = self.phase15_attention(x=pc_f_feats2.permute(0, 2, 1))
                img_f_feats2 = self.phase15_attention(x=img_f_feats2.permute(0, 2, 1))
                pc_aggr_feat_2 = self.phase15_aggregator2(pc_f_feats2.permute(0, 2, 1))
                img_aggr_feat_2 = self.phase15_aggregator2(img_f_feats2.permute(0, 2, 1))
            elif self.cfgs.phase15_attention_type == 4:
                f_img_size = img_f_feats2.shape[2:4]
                img_f_feats2 = img_f_feats2.reshape(B, -1, f_img_size[0] * f_img_size[1])
                pc_f_feats2 = self.phase15_attention(x=pc_f_feats2.permute(0, 2, 1))
                img_f_feats2 = self.phase15_attention(x=img_f_feats2.permute(0, 2, 1))
                pc_aggr_feat_2 = self.phase15_aggregator2(pc_f_feats2.permute(0, 2, 1))
                img_aggr_feat_2 = self.phase15_aggregator2(img_f_feats2.permute(0, 2, 1))
            if self.training and self.cfgs.phase15_feature_train_type == 'split':
                data_output['embeddings2'] = pc_aggr_feat
                data_output['embeddings1'] = img_aggr_feat
                data_output['embeddings4'] = pc_aggr_feat_2
                data_output['embeddings3'] = img_aggr_feat_2
            elif self.cfgs.phase15_feature_train_type == 'concat':
                data_output['embeddings2'] = torch.cat([pc_aggr_feat, pc_aggr_feat_2], dim=-1)
                data_output['embeddings1'] = torch.cat([img_aggr_feat, img_aggr_feat_2], dim=-1)
            else:
                data_output['embeddings2'] = pc_aggr_feat
                data_output['embeddings1'] = img_aggr_feat
            if self.training:
                overlap_ratio_matrix_inuse, img_pair_embeddings, pc_pair_embeddings = generate_single_correspondence_phase4_v2(points_local_inuse, 
                                                data_dict, 
                                                img_H, 
                                                img_W, 
                                                device, 
                                                img_feats_local_inuse, 
                                                pc_feats_local_inuse, 
                                                self.cfgs)
                data_output['img_local_embeddings1'] = img_pair_embeddings
                data_output['pc_local_embeddings1'] = pc_pair_embeddings
                data_output['local_overlap_ratio1'] = overlap_ratio_matrix_inuse
        elif self.cfgs.phase == 16:
            if 'phase16_feats_length' in self.cfgs.keys() and self.cfgs.phase16_feats_length == 3:
                img_feats_list = [img_2_feats, img_3_feats, img_4_feats]
                pc_feats_list = [pc_2_feats, pc_3_feats, pc_4_feats]
                pc_coords_list = [points_2, points_3, points_4]
                feats_length = 3
            else:
                img_feats_list = [img_1_feats, img_2_feats, img_3_feats, img_4_feats]
                pc_feats_list = [pc_1_feats, pc_2_feats, pc_3_feats, pc_4_feats]
                pc_coords_list = [points_1, points_2, points_3, points_4]
                feats_length = 4

            if 'image_paths' in data_dict.keys():
                dir_to_save = '/home/pengjianyi/code_projects/vis_filter_feats_0823'
                os.makedirs(dir_to_save, exist_ok=True)
                to_save_dict = {}
                to_save_dict['image_paths'] = data_dict['image_paths']
            
            if self.cfgs.img_attn_re_cfgs.to_aggr_feat_select_type == 1:
                img_aggr_feat_list, img_to_aggr_feat_idx_list = self.img_attn_re(img_feats_list)
                img_to_aggr_feat_list = []
                for i in range(feats_length):
                    curr_img_to_aggr_feat = torch.gather(img_feats_list[i].flatten(2),
                                                         dim=-1,
                                                         index=img_to_aggr_feat_idx_list[i][0].unsqueeze(1).expand(-1, self.out_dim, -1))
                    img_to_aggr_feat_list.append(curr_img_to_aggr_feat)
                img_to_aggr_feat = torch.cat(img_to_aggr_feat_list, dim=-1)
                img_aggr_feat1 = img_aggr_feat_list[0]
                img_aggr_feat2 = self.phase16_aggregator(img_to_aggr_feat)
                img_aggr_feat = torch.cat([img_aggr_feat1, img_aggr_feat2], dim=-1)
            elif self.cfgs.img_attn_re_cfgs.to_aggr_feat_select_type == 2:
                img_aggr_feat_list, img_to_aggr_feat_idx_list = self.img_attn_re(img_feats_list)
                new_img_to_aggr_feat_idx_list = []
                for i in range(feats_length):
                    curr_img_to_aggr_feat_idx = torch.cat(img_to_aggr_feat_idx_list[i], dim=-1)
                    new_img_to_aggr_feat_idx_list.append(curr_img_to_aggr_feat_idx)
                
                if 'image_paths' in data_dict.keys():
                    img_to_save_feat_idx_list = []
                    for i in range(feats_length):
                        curr_img_to_aggr_feat_idx = new_img_to_aggr_feat_idx_list[i]
                        curr_img_to_aggr_feat_idx = curr_img_to_aggr_feat_idx.detach().cpu().numpy()
                        img_to_save_feat_idx_list.append(curr_img_to_aggr_feat_idx)
                    to_save_dict['img_to_aggr_feat_idx_list'] = img_to_save_feat_idx_list
                
                img_to_aggr_feat_list = []
                for i in range(feats_length):
                    curr_img_to_aggr_feat = torch.gather(img_feats_list[i].flatten(2),
                                                         dim=-1,
                                                         index=new_img_to_aggr_feat_idx_list[i].unsqueeze(1).expand(-1, self.out_dim, -1))
                    img_to_aggr_feat_list.append(curr_img_to_aggr_feat)
                for i in range(feats_length):
                    curr_img_to_aggr_feat = img_to_aggr_feat_list[i]
                    curr_img_aggr_feat = getattr(self, f'phase16_aggregator_{i+1}')(curr_img_to_aggr_feat)
                    img_aggr_feat_list.append(curr_img_aggr_feat)
                img_aggr_feat = torch.cat(img_aggr_feat_list, dim=-1)
            elif self.cfgs.img_attn_re_cfgs.to_aggr_feat_select_type == 3:
                img_aggr_feat_list, img_to_aggr_feat_list = self.img_attn_re(img_feats_list)
                img_aggr_feat1 = img_aggr_feat_list[0]
                img_to_aggr_feat = torch.cat(img_to_aggr_feat_list, dim=1)
                img_to_aggr_feat = img_to_aggr_feat.permute(0, 2, 1)
                img_aggr_feat2 = self.phase16_aggregator(img_to_aggr_feat)
                img_aggr_feat = torch.cat([img_aggr_feat1, img_aggr_feat2], dim=-1)
            else:
                raise NotImplementedError
            
            if self.cfgs.pc_attn_re_cfgs.to_aggr_feat_select_type == 1:
                pc_aggr_feat_list, pc_to_aggr_feat_idx_list = self.pc_attn_re(pc_feats_list, pc_coords_list)
                pc_to_aggr_feat_list = []
                for i in range(feats_length):
                    curr_pc_to_aggr_feat = torch.gather(pc_feats_list[i],
                                                         dim=-1,
                                                         index=pc_to_aggr_feat_idx_list[i][0].unsqueeze(1).expand(-1, self.out_dim, -1))
                    pc_to_aggr_feat_list.append(curr_pc_to_aggr_feat)
                pc_to_aggr_feat = torch.cat(pc_to_aggr_feat_list, dim=-1)
                pc_aggr_feat1 = pc_aggr_feat_list[0]
                pc_aggr_feat2 = self.phase16_aggregator(pc_to_aggr_feat)
                pc_aggr_feat = torch.cat([pc_aggr_feat1, pc_aggr_feat2], dim=-1)
            elif self.cfgs.pc_attn_re_cfgs.to_aggr_feat_select_type == 2:
                pc_aggr_feat_list, pc_to_aggr_feat_idx_list = self.pc_attn_re(pc_feats_list, pc_coords_list)
                new_pc_to_aggr_feat_idx_list = []
                for i in range(feats_length):
                    curr_pc_to_aggr_feat_idx = torch.cat(pc_to_aggr_feat_idx_list[i], dim=-1)
                    new_pc_to_aggr_feat_idx_list.append(curr_pc_to_aggr_feat_idx)

                if 'image_paths' in data_dict.keys():
                    to_save_dict['pc_points'] = data_dict['clouds'].detach().cpu().numpy()
                    pc_to_save_feat_idx_list = []
                    for i in range(feats_length):
                        curr_pc_to_aggr_feat_idx = new_pc_to_aggr_feat_idx_list[i]
                        curr_pc_to_aggr_feat_idx = curr_pc_to_aggr_feat_idx.detach().cpu().numpy()
                        pc_to_save_feat_idx_list.append(curr_img_to_aggr_feat_idx)
                    to_save_dict['pc_to_aggr_feat_idx_list'] = pc_to_save_feat_idx_list

                pc_to_aggr_feat_list = []
                for i in range(feats_length):
                    curr_pc_to_aggr_feat = torch.gather(pc_feats_list[i],
                                                         dim=-1,
                                                         index=new_pc_to_aggr_feat_idx_list[i].unsqueeze(1).expand(-1, self.out_dim, -1))
                    pc_to_aggr_feat_list.append(curr_pc_to_aggr_feat)
                for i in range(feats_length):
                    curr_pc_to_aggr_feat = pc_to_aggr_feat_list[i]
                    curr_pc_aggr_feat = getattr(self, f'phase16_aggregator_{i+1}')(curr_pc_to_aggr_feat)
                    pc_aggr_feat_list.append(curr_pc_aggr_feat)
                pc_aggr_feat = torch.cat(pc_aggr_feat_list, dim=-1)
            elif self.cfgs.img_attn_re_cfgs.to_aggr_feat_select_type == 3:
                pc_aggr_feat_list, pc_to_aggr_feat_list = self.pc_attn_re(pc_feats_list, pc_coords_list)
                pc_aggr_feat1 = pc_aggr_feat_list[0]
                pc_to_aggr_feat = torch.cat(pc_to_aggr_feat_list, dim=1)
                pc_to_aggr_feat = pc_to_aggr_feat.permute(0, 2, 1)
                pc_aggr_feat2 = self.phase16_aggregator(pc_to_aggr_feat)
                pc_aggr_feat = torch.cat([pc_aggr_feat1, pc_aggr_feat2], dim=-1)
            else:
                raise NotImplementedError
            
            if 'image_paths' in data_dict.keys():
                bn_for_filter = int(data_dict['bn_for_filter'])
                path_to_save = os.path.join(dir_to_save, f'{bn_for_filter}.pkl')
                with open(path_to_save, 'wb') as f:
                    pickle.dump(to_save_dict, f)

            data_output['embeddings2'] = pc_aggr_feat
            data_output['embeddings1'] = img_aggr_feat
        elif self.cfgs.phase == 17:
            if self.cfgs.phase17_use_four_aggregator:
                if self.cfgs.phase17_use_SA_block:
                    img_feats_inuse = img_1_feats.flatten(2).permute(2, 0, 1)
                    pc_feats_inuse = pc_1_feats.permute(2, 0, 1)
                    img_cls_token = self.phase17_img_cls_emb.expand(-1, B, -1) # [1, B, C]
                    pc_cls_token = self.phase17_pc_cls_emb.expand(-1, B, -1) # [1, B, C]
                    img_feats_inuse = torch.cat([img_cls_token, img_feats_inuse], dim=0) # [(H*W)+1, B, C]
                    pc_feats_inuse = torch.cat([pc_cls_token, pc_feats_inuse], dim=0) # [N+1, B, C]
                    img_feats_inuse = img_feats_inuse + self.phase17_img_pos_emb.permute(1, 0, 2) # [(H*W)+1, B, C]
                    if self.cfgs.phase17_pc_use_pos_emb:
                        pc_feats_inuse = pc_feats_inuse + self.phase17_pc_pos_emb.permute(1, 0, 2) # [N+1, B, C]
                    for i in range(len(self.attn_img)):
                        img_feat_inuse, _ = self.attn_img[i](img_feats_inuse) # [(H*W)+1, B, C]、[B, (H*W)+1, (H*W)+1]
                    for i in range(len(self.attn_pc)):
                        pc_feat_inuse, _ = self.attn_pc[i](pc_feats_inuse) # [N+1, B, C]、[B, N+1, N+1]
                    if self.cfgs.phase17_SA_block_aggr_type == 1:
                        img_feat_inuse = img_feat_inuse[0, :, :] # [B, C]
                        pc_feat_inuse = pc_feat_inuse[0, :, :] # [B, C]
                    elif self.cfgs.phase17_SA_block_aggr_type == 2:
                        img_feat_inuse = self.phase17_SA_block_aggregator(img_feat_inuse[1:, ...].permute(1, 2, 0))
                        pc_feat_inuse = self.phase17_SA_block_aggregator(pc_feat_inuse[1:, ...].permute(1, 2, 0))
                else:
                    img_feat_inuse = self.phase17_aggregator(img_1_feats)
                    pc_feat_inuse = self.phase17_aggregator(pc_1_feats)

                img_feat_inuse_1 = self.phase17_aggregator_1(img_1_feats)
                pc_feat_inuse_1 = self.phase17_aggregator_1(pc_1_feats)
                img_feat_inuse_2 = self.phase17_aggregator_2(img_2_feats)
                pc_feat_inuse_2 = self.phase17_aggregator_2(pc_2_feats)
                img_feat_inuse_3 = self.phase17_aggregator_3(img_3_feats)
                pc_feat_inuse_3 = self.phase17_aggregator_3(pc_3_feats)
                img_feat_inuse_4 = self.phase17_aggregator_4(img_4_feats)
                pc_feat_inuse_4 = self.phase17_aggregator_4(pc_4_feats)

                if 'phase17_feat_fuse_type' in self.cfgs.keys():
                    phase17_feat_fuse_type = self.cfgs.phase17_feat_fuse_type

                if phase17_feat_fuse_type == 'concat':
                    img_feat_inuse = torch.cat([img_feat_inuse, img_feat_inuse_1, img_feat_inuse_2, img_feat_inuse_3, img_feat_inuse_4], dim=-1)
                    pc_feat_inuse = torch.cat([pc_feat_inuse, pc_feat_inuse_1, pc_feat_inuse_2, pc_feat_inuse_3, pc_feat_inuse_4], dim=-1)
                    data_output['embeddings2'] = pc_feat_inuse
                    data_output['embeddings1'] = img_feat_inuse
                elif self.training and phase17_feat_fuse_type == 'split':
                    data_output['embeddings2'] = pc_feat_inuse
                    data_output['embeddings1'] = img_feat_inuse
                    data_output['embeddings4'] = pc_feat_inuse_1
                    data_output['embeddings3'] = img_feat_inuse_1
                    data_output['embeddings6'] = pc_feat_inuse_2
                    data_output['embeddings5'] = img_feat_inuse_2
                    data_output['embeddings8'] = pc_feat_inuse_3
                    data_output['embeddings7'] = img_feat_inuse_3
                    data_output['embeddings10'] = pc_feat_inuse_4
                    data_output['embeddings9'] = img_feat_inuse_4
                else:
                    data_output['embeddings2'] = pc_feat_inuse
                    data_output['embeddings1'] = img_feat_inuse
            else:
                if self.cfgs.phase17_use_SA_block:
                    if self.cfgs.image_out_layer is not None:
                        img_feats_inuse = img_feats_inuse.flatten(2).permute(2, 0, 1)
                        pc_feats_inuse = pc_feats_inuse.permute(2, 0, 1)
                    else:
                        img_feats_inuse = c_img_feats.flatten(2).permute(2, 0, 1)
                        pc_feats_inuse = c_pc_feats.permute(2, 0, 1)
                    img_cls_token = self.phase17_img_cls_emb.expand(-1, B, -1) # [1, B, C]
                    pc_cls_token = self.phase17_pc_cls_emb.expand(-1, B, -1) # [1, B, C]
                    img_feats_inuse = torch.cat([img_cls_token, img_feats_inuse], dim=0) # [(H*W)+1, B, C]
                    pc_feats_inuse = torch.cat([pc_cls_token, pc_feats_inuse], dim=0) # [N+1, B, C]
                    img_feats_inuse = img_feats_inuse + self.phase17_img_pos_emb.permute(1, 0, 2) # [(H*W)+1, B, C]
                    if self.cfgs.phase17_pc_use_pos_emb:
                        pc_feats_inuse = pc_feats_inuse + self.phase17_pc_pos_emb.permute(1, 0, 2) # [N+1, B, C]
                    for i in range(len(self.attn_img)):
                        img_feat_inuse, img_attn = self.attn_img[i](img_feats_inuse) # [(H*W)+1, B, C]、[B, (H*W)+1, (H*W)+1]
                    for i in range(len(self.attn_pc)):
                        pc_feat_inuse, pc_attn = self.attn_pc[i](pc_feats_inuse) # [N+1, B, C]、[B, N+1, N+1]
                    if self.cfgs.phase17_SA_block_aggr_type == 1:
                        img_feat_inuse = img_feat_inuse[0, :, :] # [B, C]
                        pc_feat_inuse = pc_feat_inuse[0, :, :] # [B, C]
                    elif self.cfgs.phase17_SA_block_aggr_type == 2:
                        img_feat_inuse = self.phase17_SA_block_aggregator(img_feat_inuse[1:, ...].permute(1, 2, 0))
                        pc_feat_inuse = self.phase17_SA_block_aggregator(pc_feat_inuse[1:, ...].permute(1, 2, 0))
                    elif self.cfgs.phase17_SA_block_aggr_type == 3:
                        img_feat_to_aggr = img_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, H*W, C]
                        img_attn_to_aggr = img_attn[:, 0, 1:] # [B, (H*W)]
                        img_feat_inuse = torch.matmul(img_attn_to_aggr.unsqueeze(1), img_feat_to_aggr).squeeze(1) # [B, C]
                        pc_feat_to_aggr = pc_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, N, C]
                        pc_attn_to_aggr = pc_attn[:, 0, 1:] # [B, N]
                        pc_feat_inuse = torch.matmul(pc_attn_to_aggr.unsqueeze(1), pc_feat_to_aggr).squeeze(1) # [B, C]

                        if 'vis_attention' in data_dict.keys() and data_dict['vis_attention']:
                            to_save_dict = {}
                            to_save_dict['image_paths'] = data_dict['image_paths']
                            img_attn_vis = img_attn_to_aggr.reshape(B, 7, 7)
                            img_attn_vis = img_attn_vis.to('cpu').detach().clone().numpy()
                            to_save_dict['img_attn_vis'] = img_attn_vis
                            bn_for_filter = int(data_dict['bn_for_filter'])
                            output_dir = data_dict['output_dir']
                            out_put_path = os.path.join(output_dir, f'{bn_for_filter}.pkl')
                            with open(out_put_path, 'wb') as f:
                                pickle.dump(to_save_dict, f)

                    elif self.cfgs.phase17_SA_block_aggr_type == 4:
                        img_feat_to_aggr = img_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, H*W, C]
                        img_attn_to_aggr = img_attn[:, 0, 1:] # [B, (H*W)]
                        img_feat_to_aggr = img_attn_to_aggr.unsqueeze(-1) * img_feat_to_aggr # [B, H*W, C]
                        pc_feat_to_aggr = pc_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, N, C]
                        pc_attn_to_aggr = pc_attn[:, 0, 1:] # [B, N]
                        pc_feat_to_aggr = pc_attn_to_aggr.unsqueeze(-1) * pc_feat_to_aggr # [B, N, C]
                        img_feat_inuse = self.phase17_SA_block_aggregator(img_feat_to_aggr.permute(0, 2, 1))
                        pc_feat_inuse = self.phase17_SA_block_aggregator(pc_feat_to_aggr.permute(0, 2, 1))
                    elif self.cfgs.phase17_SA_block_aggr_type == 5:
                        img_feat_to_aggr = img_feat_inuse[1:, :, :].permute(1, 2, 0) # [B, C, H*W]
                        img_feat_inuse = F.adaptive_avg_pool1d(img_feat_to_aggr, 1).squeeze(-1)
                        pc_feat_to_aggr = pc_feat_inuse[1:, :, :].permute(1, 2, 0) # [B, C, N]
                        pc_feat_inuse = F.adaptive_avg_pool1d(pc_feat_to_aggr, 1).squeeze(-1)
                    elif self.cfgs.phase17_SA_block_aggr_type == 6:
                        img_feat_to_aggr = img_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, H*W, C]
                        img_attn_to_aggr = img_attn[:, 0, 1:] # [B, (H*W)]
                        img_attn_to_aggr_sum = img_attn_to_aggr.sum(dim=-1, keepdim=True)
                        img_feat_inuse = torch.matmul(img_attn_to_aggr.unsqueeze(1), img_feat_to_aggr).squeeze(1) # [B, C]
                        img_feat_inuse = img_feat_inuse / (img_attn_to_aggr_sum + 1e-6)
                        pc_feat_to_aggr = pc_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, N, C]
                        pc_attn_to_aggr = pc_attn[:, 0, 1:] # [B, N]
                        pc_attn_to_aggr_sum = pc_attn_to_aggr.sum(dim=-1, keepdim=True)
                        pc_feat_inuse = torch.matmul(pc_attn_to_aggr.unsqueeze(1), pc_feat_to_aggr).squeeze(1) # [B, C]
                        pc_feat_inuse = pc_feat_inuse / (pc_attn_to_aggr_sum + 1e-6)
                    data_output['embeddings2'] = pc_feat_inuse
                    data_output['embeddings1'] = img_feat_inuse
                else:
                    img_feat_inuse = self.phase17_aggregator(c_img_feats)
                    pc_feat_inuse = self.phase17_aggregator(c_pc_feats)
                    data_output['embeddings2'] = pc_feat_inuse
                    data_output['embeddings1'] = img_feat_inuse

        elif self.cfgs.phase == 18:
            if self.cfgs.phase18_pc_aggr_layer == 1:
                pc_feats_aggr_inuse = c_pc_feats
            elif self.cfgs.phase18_pc_aggr_layer == 2:
                pc_feats_aggr_inuse = f_pc_feats
            elif self.cfgs.phase18_pc_aggr_layer == 3:
                pc_feats_aggr_inuse = pc_c_feats
            elif self.cfgs.phase18_pc_aggr_layer == 4:
                pc_feats_aggr_inuse = pc_f_feats2
            if self.cfgs.phase18_img_aggr_layer == 1:
                img_feats_aggr_inuse = c_img_feats
            elif self.cfgs.phase18_img_aggr_layer == 2:
                img_feats_aggr_inuse = f_img_feats
            elif self.cfgs.phase18_img_aggr_layer == 3:
                img_feats_aggr_inuse = img_c_feats
            elif self.cfgs.phase18_img_aggr_layer == 4:
                img_feats_aggr_inuse = img_f_feats2
            if self.cfgs.phase18_pc_local_layer == 1:
                pc_feats_local_inuse = c_pc_feats
                points_local_inuse = c_points
            elif self.cfgs.phase18_pc_local_layer == 2:
                pc_feats_local_inuse = f_pc_feats
                points_local_inuse = f_points
            elif self.cfgs.phase18_pc_local_layer == 3:
                pc_feats_local_inuse = pc_f_feats1
                points_local_inuse = f_points
            if self.cfgs.phase18_img_local_layer == 1:
                img_feats_local_inuse = c_img_feats
            elif self.cfgs.phase18_img_local_layer == 2:
                img_feats_local_inuse = f_img_feats
            elif self.cfgs.phase18_img_local_layer == 3:
                img_feats_local_inuse = img_f_feats1
                data_output['rgb_depth_preds'] = img_f_feats2.squeeze(1)
            elif self.cfgs.phase18_img_local_layer == 4:
                img_feats_local_inuse = img_f_feats1
            pc_aggr_feat = self.phase18_aggregator(pc_feats_aggr_inuse)
            img_aggr_feat = self.phase18_aggregator(img_feats_aggr_inuse)

            # TODO: instead of GeM, use concatenate and go though a MLP to aggregate semantic features
            if 'aggregate_clusterly_and_semantically' in self.cfgs.keys():
                if 'aggregate_and_match' in self.cfgs.keys() and  self.cfgs.aggregate_and_match:
                    (img_semantic_embeddings, 
                    img_semantic_mask, 
                    pc_semantic_embeddings, 
                    pc_semantic_mask, 
                    img_cluster_embeddings, 
                    img_cluster_masks, 
                    pc_cluster_embeddings, 
                    pc_cluster_masks,
                    pc_cluster_out_embeddings,
                    img_cluster_out_embeddings,
                    pc_semantic_out_embeddings,
                    img_semantic_out_embeddings,
                    pc_all_semantic_out_embeddings,
                    img_all_semantic_out_embeddings)= aggregate_and_match(data_dict, 
                                                    device, 
                                                    img_feats_local_inuse, 
                                                    pc_feats_local_inuse, 
                                                    self.cfgs, 
                                                    self.cluster_transformer, 
                                                    self.semantic_transformer,
                                                    points_local_inuse,
                                                    'train' if self.training else 'eval')
                    data_output['img_cluster_embeddings'] = img_cluster_out_embeddings
                    data_output['pc_cluster_embeddings'] = pc_cluster_out_embeddings
                    data_output['img_in_batch_semantic_embeddings'] = img_semantic_out_embeddings
                    data_output['pc_in_batch_semantic_embeddings'] = pc_semantic_out_embeddings
                    data_output['img_all_semantic_embeddings'] = img_all_semantic_out_embeddings
                    data_output['pc_all_semantic_embeddings'] = pc_all_semantic_out_embeddings
                else:
                    (img_semantic_embeddings, 
                    img_semantic_mask, 
                    pc_semantic_embeddings, 
                    pc_semantic_mask, 
                    img_cluster_embeddings, 
                    img_cluster_masks, 
                    pc_cluster_embeddings, 
                    pc_cluster_masks)= aggregate_clusterly_and_semantically(data_dict, 
                                                        device, 
                                                        img_feats_local_inuse, 
                                                        pc_feats_local_inuse, 
                                                        self.cfgs, 
                                                        self.cluster_transformer, 
                                                        self.semantic_transformer)
                if self.cfgs.aggregate_clusterly_and_semantically == 'only_clusterly':
                    img_cluster_aggr_feat = self.cluster_aggregator(img_cluster_embeddings, img_cluster_masks)
                    pc_cluster_aggr_feat = self.cluster_aggregator(pc_cluster_embeddings, pc_cluster_masks)
                    if self.cfgs.use_ordinary_aggregator:
                        img_aggr_feat = torch.cat((img_aggr_feat, img_cluster_aggr_feat), dim=-1)
                        pc_aggr_feat = torch.cat((pc_aggr_feat, pc_cluster_aggr_feat), dim=-1)
                    else:
                        pc_aggr_feat = pc_cluster_aggr_feat 
                        img_aggr_feat = img_cluster_aggr_feat
                elif self.cfgs.aggregate_clusterly_and_semantically == 'only_semantically':
                    img_semantic_aggr_feat = self.semantic_aggregator(img_semantic_embeddings, img_semantic_mask)
                    pc_semantic_aggr_feat = self.semantic_aggregator(pc_semantic_embeddings, pc_semantic_mask)
                    if self.cfgs.use_ordinary_aggregator:
                        img_aggr_feat = torch.cat((img_aggr_feat, img_semantic_aggr_feat), dim=-1)
                        pc_aggr_feat = torch.cat((pc_aggr_feat, pc_semantic_aggr_feat), dim=-1)
                    else:
                        pc_aggr_feat = pc_semantic_aggr_feat 
                        img_aggr_feat = img_semantic_aggr_feat
                elif self.cfgs.aggregate_clusterly_and_semantically == 'both':
                    img_cluster_aggr_feat = self.cluster_aggregator(img_cluster_embeddings, img_cluster_masks)
                    pc_cluster_aggr_feat = self.cluster_aggregator(pc_cluster_embeddings, pc_cluster_masks)
                    img_semantic_aggr_feat = self.semantic_aggregator(img_semantic_embeddings, img_semantic_mask)
                    pc_semantic_aggr_feat = self.semantic_aggregator(pc_semantic_embeddings, pc_semantic_mask)
                    if self.cfgs.use_ordinary_aggregator:
                        img_aggr_feat = torch.cat((img_aggr_feat, img_cluster_aggr_feat, img_semantic_aggr_feat), dim=-1)
                        pc_aggr_feat = torch.cat((pc_aggr_feat, pc_cluster_aggr_feat, pc_semantic_aggr_feat), dim=-1)
                    else:
                        pc_aggr_feat = torch.cat((pc_cluster_aggr_feat, pc_semantic_aggr_feat), dim=-1)
                        img_aggr_feat = torch.cat((img_cluster_aggr_feat, img_semantic_aggr_feat), dim=-1)
            
            data_output['embeddings2'] = pc_aggr_feat
            data_output['embeddings1'] = img_aggr_feat

            if self.training:
                if ('phase18_local_correspondence' not in self.cfgs.keys()) or self.cfgs.phase18_local_correspondence:
                    (cluster_overlap_ratio_choose, 
                    img_cluster_embeddings, 
                    pc_cluster_embeddings, 
                    img_in_batch_semantic_embeddings, 
                    pc_in_batch_semantic_embeddings, 
                    img_semantic_embeddings, 
                    pc_semantic_embeddings) = generate_cluster_correspondence(points_local_inuse, 
                                                    data_dict, 
                                                    img_H, 
                                                    img_W, 
                                                    device, 
                                                    img_feats_local_inuse, 
                                                    pc_feats_local_inuse, 
                                                    self.cfgs)
                    if 'phase18_attention_embeddings' in self.cfgs.keys() and self.cfgs.phase18_attention_embeddings:
                        pass
                        # TODO: try several kinds of attention mechanism only after current phase is useful

                    # TODO: generate positive and negative masks to distinguish "same semantic different cluster"、"same all semantic different batch", incase of using infonce loss
                    if 'random_shuffle' in self.cfgs.keys() and self.cfgs.random_shuffle:
                        img_cluster_embeddings = img_cluster_embeddings[torch.randperm(img_cluster_embeddings.shape[0])]
                        img_in_batch_semantic_embeddings = img_in_batch_semantic_embeddings[torch.randperm(img_in_batch_semantic_embeddings.shape[0])]
                        img_semantic_embeddings = img_semantic_embeddings[torch.randperm(img_semantic_embeddings.shape[0])]

                    data_output['cluster_overlap_ratio_choose'] = cluster_overlap_ratio_choose
                    data_output['img_cluster_embeddings'] = img_cluster_embeddings
                    data_output['pc_cluster_embeddings'] = pc_cluster_embeddings
                    data_output['img_in_batch_semantic_embeddings'] = img_in_batch_semantic_embeddings
                    data_output['pc_in_batch_semantic_embeddings'] = pc_in_batch_semantic_embeddings
                    data_output['img_all_semantic_embeddings'] = img_semantic_embeddings
                    data_output['pc_all_semantic_embeddings'] = pc_semantic_embeddings
        elif self.cfgs.phase == 19:
            if self.cfgs.phase19_aggregator1_type == 'PoS_GeM':
                pc_coords_list = [points_4, points_3, points_2, points_1]
                pc_index_list, pc_coords_list = generate_pc_index_and_coords_v2(pc_coords_list)
                img_index_list, _, img_coords_list = generate_img_index_and_knn_and_coords_v3(B, self.cfgs.phase19_resolution_list, device)
                if self.cfgs.phase19_modal_split_aggregator:
                    pc_aggr_feat_1 = self.phase19_aggregator1_1(pc_1_feats, mask=None, index_list=pc_index_list[-2:], coords_list=pc_coords_list[-2:])
                    img_1_feats = torch.flatten(img_1_feats, start_dim=2)
                    img_aggr_feat_1 = self.phase19_aggregator1_2(img_1_feats, mask=None, index_list=img_index_list[-2:], coords_list=img_coords_list[-2:])
                    pc_aggr_feat_2 = self.phase19_aggregator2_1(pc_2_feats, mask=None, index_list=pc_index_list[-3:], coords_list=pc_coords_list[-3:])
                    img_2_feats = torch.flatten(img_2_feats, start_dim=2)
                    img_aggr_feat_2 = self.phase19_aggregator2_2(img_2_feats, mask=None, index_list=img_index_list[-3:], coords_list=img_coords_list[-3:])
                    pc_aggr_feat_3 = self.phase19_aggregator3_1(pc_3_feats, mask=None, index_list=pc_index_list[-4:], coords_list=pc_coords_list[-4:])
                    img_3_feats = torch.flatten(img_3_feats, start_dim=2)
                    img_aggr_feat_3 = self.phase19_aggregator3_2(img_3_feats, mask=None, index_list=img_index_list[-4:], coords_list=img_coords_list[-4:])
                    pc_aggr_feat_4 = self.phase19_aggregator4_1(pc_4_feats, mask=None, index_list=pc_index_list[-5:], coords_list=pc_coords_list[-5:])
                    img_4_feats = torch.flatten(img_4_feats, start_dim=2)
                    img_aggr_feat_4 = self.phase19_aggregator4_2(img_4_feats, mask=None, index_list=img_index_list[-5:], coords_list=img_coords_list[-5:])
                else:
                    pc_aggr_feat_1 = self.phase19_aggregator1(pc_1_feats, mask=None, index_list=pc_index_list[-2:], coords_list=pc_coords_list[-2:])
                    img_1_feats = torch.flatten(img_1_feats, start_dim=2)
                    img_aggr_feat_1 = self.phase19_aggregator1(img_1_feats, mask=None, index_list=img_index_list[-2:], coords_list=img_coords_list[-2:])
                    pc_aggr_feat_2 = self.phase19_aggregator2(pc_2_feats, mask=None, index_list=pc_index_list[-3:], coords_list=pc_coords_list[-3:])
                    img_2_feats = torch.flatten(img_2_feats, start_dim=2)
                    img_aggr_feat_2 = self.phase19_aggregator2(img_2_feats, mask=None, index_list=img_index_list[-3:], coords_list=img_coords_list[-3:])
                    pc_aggr_feat_3 = self.phase19_aggregator3(pc_3_feats, mask=None, index_list=pc_index_list[-4:], coords_list=pc_coords_list[-4:])
                    img_3_feats = torch.flatten(img_3_feats, start_dim=2)
                    img_aggr_feat_3 = self.phase19_aggregator3(img_3_feats, mask=None, index_list=img_index_list[-4:], coords_list=img_coords_list[-4:])
                    pc_aggr_feat_4 = self.phase19_aggregator4(pc_4_feats, mask=None, index_list=pc_index_list[-5:], coords_list=pc_coords_list[-5:])
                    img_4_feats = torch.flatten(img_4_feats, start_dim=2)
                    img_aggr_feat_4 = self.phase19_aggregator4(img_4_feats, mask=None, index_list=img_index_list[-5:], coords_list=img_coords_list[-5:])
            elif self.cfgs.phase19_aggregator1_type == 'GeM':
                if self.cfgs.phase19_modal_split_aggregator:
                    pc_aggr_feat_1 = self.phase19_aggregator1_1(pc_1_feats)
                    img_aggr_feat_1 = self.phase19_aggregator1_2(img_1_feats)
                    pc_aggr_feat_2 = self.phase19_aggregator2_1(pc_2_feats)
                    img_aggr_feat_2 = self.phase19_aggregator2_2(img_2_feats)
                    pc_aggr_feat_3 = self.phase19_aggregator3_1(pc_3_feats)
                    img_aggr_feat_3 = self.phase19_aggregator3_2(img_3_feats)
                    pc_aggr_feat_4 = self.phase19_aggregator4_1(pc_4_feats)
                    img_aggr_feat_4 = self.phase19_aggregator4_2(img_4_feats)
                else:
                    pc_aggr_feat_1 = self.phase19_aggregator1(pc_1_feats)
                    img_aggr_feat_1 = self.phase19_aggregator1(img_1_feats)
                    pc_aggr_feat_2 = self.phase19_aggregator2(pc_2_feats)
                    img_aggr_feat_2 = self.phase19_aggregator2(img_2_feats)
                    pc_aggr_feat_3 = self.phase19_aggregator3(pc_3_feats)
                    img_aggr_feat_3 = self.phase19_aggregator3(img_3_feats)
                    pc_aggr_feat_4 = self.phase19_aggregator4(pc_4_feats)
                    img_aggr_feat_4 = self.phase19_aggregator4(img_4_feats)
            else:
                raise NotImplementedError

            data_output['embeddings2'] = torch.cat([pc_aggr_feat_1, pc_aggr_feat_2, pc_aggr_feat_3, pc_aggr_feat_4], dim=-1)
            data_output['embeddings1'] = torch.cat([img_aggr_feat_1, img_aggr_feat_2, img_aggr_feat_3, img_aggr_feat_4], dim=-1)
        elif self.cfgs.phase == 20:
            img_feats_inuse = c_img_feats.flatten(2).permute(2, 0, 1)
            pc_feats_inuse = c_pc_feats.permute(2, 0, 1)
            img_cls_token = self.phase20_img_cls_emb.expand(-1, B, -1) # [1, B, C]
            pc_cls_token = self.phase20_pc_cls_emb.expand(-1, B, -1) # [1, B, C]
            img_feats_inuse = torch.cat([img_cls_token, img_feats_inuse], dim=0) # [(H*W)+1, B, C]
            pc_feats_inuse = torch.cat([pc_cls_token, pc_feats_inuse], dim=0) # [N+1, B, C]
            for i in range(len(self.phase20_attention)):
                img_feat_inuse, img_attn = self.phase20_attention[i](img_feats_inuse) # [(H*W)+1, B, C]、[B, (H*W)+1, (H*W)+1]
                pc_feat_inuse, pc_attn = self.phase20_attention[i](pc_feats_inuse) # [N+1, B, C]、[B, N+1, N+1]
            if self.cfgs.phase20_SA_block_aggr_type == 1:
                img_feat_inuse = img_feat_inuse[0, :, :] # [B, C]
                pc_feat_inuse = pc_feat_inuse[0, :, :] # [B, C]
            elif self.cfgs.phase20_SA_block_aggr_type == 2:
                img_feat_inuse = self.phase20_SA_block_aggregator(img_feat_inuse[1:, ...].permute(1, 2, 0))
                pc_feat_inuse = self.phase20_SA_block_aggregator(pc_feat_inuse[1:, ...].permute(1, 2, 0))
            elif self.cfgs.phase20_SA_block_aggr_type == 3:
                img_feat_to_aggr = img_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, H*W, C]
                img_attn_to_aggr = img_attn[:, 0, 1:] # [B, (H*W)]
                img_feat_inuse = torch.matmul(img_attn_to_aggr.unsqueeze(1), img_feat_to_aggr).squeeze(1) # [B, C]
                pc_feat_to_aggr = pc_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, N, C]
                pc_attn_to_aggr = pc_attn[:, 0, 1:] # [B, N]
                pc_feat_inuse = torch.matmul(pc_attn_to_aggr.unsqueeze(1), pc_feat_to_aggr).squeeze(1) # [B, C]

                if 'vis_attention' in data_dict.keys() and data_dict['vis_attention']:
                    to_save_dict = {}
                    to_save_dict['image_paths'] = data_dict['image_paths']
                    img_attn_vis = img_attn_to_aggr.reshape(B, 7, 7)
                    img_attn_vis = img_attn_vis.to('cpu').detach().clone().numpy()
                    to_save_dict['img_attn_vis'] = img_attn_vis
                    bn_for_filter = int(data_dict['bn_for_filter'])
                    output_dir = data_dict['output_dir']
                    out_put_path = os.path.join(output_dir, f'{bn_for_filter}.pkl')
                    with open(out_put_path, 'wb') as f:
                        pickle.dump(to_save_dict, f)

            elif self.cfgs.phase20_SA_block_aggr_type == 4:
                img_feat_to_aggr = img_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, H*W, C]
                img_attn_to_aggr = img_attn[:, 0, 1:] # [B, (H*W)]
                img_feat_to_aggr = img_attn_to_aggr.unsqueeze(-1) * img_feat_to_aggr # [B, H*W, C]
                pc_feat_to_aggr = pc_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, N, C]
                pc_attn_to_aggr = pc_attn[:, 0, 1:] # [B, N]
                pc_feat_to_aggr = pc_attn_to_aggr.unsqueeze(-1) * pc_feat_to_aggr # [B, N, C]
                img_feat_inuse = self.phase20_SA_block_aggregator(img_feat_to_aggr.permute(0, 2, 1))
                pc_feat_inuse = self.phase20_SA_block_aggregator(pc_feat_to_aggr.permute(0, 2, 1))
            elif self.cfgs.phase20_SA_block_aggr_type == 5:
                img_feat_to_aggr = img_feat_inuse[1:, :, :].permute(1, 2, 0) # [B, C, H*W]
                img_feat_inuse = F.adaptive_avg_pool1d(img_feat_to_aggr, 1).squeeze(-1)
                pc_feat_to_aggr = pc_feat_inuse[1:, :, :].permute(1, 2, 0) # [B, C, N]
                pc_feat_inuse = F.adaptive_avg_pool1d(pc_feat_to_aggr, 1).squeeze(-1)
            elif self.cfgs.phase17_SA_block_aggr_type == 6:
                img_feat_to_aggr = img_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, H*W, C]
                img_attn_to_aggr = img_attn[:, 0, 1:] # [B, (H*W)]
                img_attn_to_aggr_sum = img_attn_to_aggr.sum(dim=-1, keepdim=True)
                img_feat_inuse = torch.matmul(img_attn_to_aggr.unsqueeze(1), img_feat_to_aggr).squeeze(1) # [B, C]
                img_feat_inuse = img_feat_inuse / (img_attn_to_aggr_sum + 1e-6)
                pc_feat_to_aggr = pc_feat_inuse[1:, :, :].permute(1, 0, 2) # [B, N, C]
                pc_attn_to_aggr = pc_attn[:, 0, 1:] # [B, N]
                pc_attn_to_aggr_sum = pc_attn_to_aggr.sum(dim=-1, keepdim=True)
                pc_feat_inuse = torch.matmul(pc_attn_to_aggr.unsqueeze(1), pc_feat_to_aggr).squeeze(1) # [B, C]
                pc_feat_inuse = pc_feat_inuse / (pc_attn_to_aggr_sum + 1e-6)
            data_output['embeddings2'] = pc_feat_inuse
            data_output['embeddings1'] = img_feat_inuse
        
        elif self.cfgs.phase == 21:
            img_feats_aggr_inuse = c_img_feats
            pc_feats_aggr_inuse = c_pc_feats


            if self.training:
                img_feats_local_inuse = f_img_feats # [B, C, H, W]
                pc_feats_local_inuse = f_pc_feats # [B, C, N]

                assert img_feats_local_inuse.shape[2] == 112
                assert img_feats_local_inuse.shape[3] == 112
                assert pc_feats_local_inuse.shape[2] == 4096

                if data_dict['pixel_selection_method'] == 'random_among_batch':
                    pixels_selected_indices = data_dict['pixels_selected_indices']
                    points_selected_indices = data_dict['points_selected_indices']
                    local_positive_mask = data_dict['local_positive_mask'].unsqueeze(0)  # produce (1, cfgs.pixel_selection_num_all, cfgs.pixel_selection_num_all * cfgs.points_selection_num)
                    local_negative_mask = data_dict['local_negative_mask'].unsqueeze(0)  # produce (1, cfgs.pixel_selection_num_all, cfgs.pixel_selection_num_all * cfgs.points_selection_num)
                    img_feats_local = img_feats_local_inuse[pixels_selected_indices[:, 0], :, pixels_selected_indices[:, 2], pixels_selected_indices[:, 1]] # (cfgs.pixel_selection_num_all, C)
                    pc_feats_local = pc_feats_local_inuse[points_selected_indices[:, 0], :, points_selected_indices[:, 1]] # (cfgs.pixel_selection_num_all * cfgs.points_selection_num, C)
                    img_feats_local = img_feats_local.unsqueeze(0) # (1, cfgs.pixel_selection_num_all, C)
                    pc_feats_local = pc_feats_local.unsqueeze(0) # (1, cfgs.pixel_selection_num_all * cfgs.points_selection_num, C)

                elif data_dict['pixel_selection_method'] == 'random_in_pair':
                    curr_img_H = img_feats_local_inuse.shape[2]
                    curr_img_W = img_feats_local_inuse.shape[3]
                    pixels_selected_indices = data_dict['pixels_selected_indices'] # produce (B, cfgs.pixel_selection_num_each_pair, 2)
                    points_selected_indices = data_dict['points_selected_indices'] # (B * B, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair)
                    local_positive_mask = data_dict['local_positive_mask'].permute(0, 2, 1) # produce (B * B, cfgs.pixel_selection_num_each_pair, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair)
                    local_negative_mask = data_dict['local_negative_mask'].permute(0, 2, 1) # produce (B * B, cfgs.pixel_selection_num_each_pair, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair)
                    pixels_selected_indices = pixels_selected_indices[:, :, 1] * curr_img_W + pixels_selected_indices[:, :, 0] # (B, cfgs.pixel_selection_num_each_pair)
                    img_feats_local = torch.gather(img_feats_local_inuse.flatten(2).permute(0, 2, 1),
                                                dim=1,
                                                index=pixels_selected_indices.unsqueeze(-1).expand(-1, -1, self.out_dim)) # (B, cfgs.pixel_selection_num_each_pair, C)
                    img_feats_local = img_feats_local.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * B, -1, self.out_dim) # (B * B, cfgs.pixel_selection_num_each_pair, C)
                    pc_feats_local = pc_feats_local_inuse.unsqueeze(1).expand(-1, B, -1, -1).flatten(0, 1).permute(0, 2, 1) # (B * B, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair, C)
                    pc_feats_local = torch.gather(pc_feats_local,
                                                    dim=1,
                                                    index=points_selected_indices.unsqueeze(-1).expand(-1, -1, self.out_dim)) # (B * B, cfgs.points_selection_num * cfgs.pixel_selection_num_each_pair, C)
                else:
                    raise NotImplementedError
                data_output['img_feats_local'] = img_feats_local
                data_output['pc_feats_local'] = pc_feats_local
                data_output['local_positive_mask'] = local_positive_mask
                data_output['local_negative_mask'] = local_negative_mask

            img_feat_inuse = self.phase21_aggregator(img_feats_aggr_inuse)
            pc_feat_inuse = self.phase21_aggregator(pc_feats_aggr_inuse)
            data_output['embeddings2'] = pc_feat_inuse
            data_output['embeddings1'] = img_feat_inuse

        elif self.cfgs.phase == 22:
            img_feats_aggr_inuse = c_img_feats
            pc_feats_aggr_inuse = c_pc_feats

            if self.training:
                img_feats_local_inuse = f_img_feats # [B, C, H, W]
                pc_feats_local_inuse = f_pc_feats # [B, C, N]
                (overlap_ratio_matrix_inuse, 
                img_pair_embeddings,
                pc_pair_embeddings,
                in_batch_flag_matrix) = generate_single_correspondence_in_pair(
                                            f_points, 
                                            data_dict, 
                                            img_H, 
                                            img_W, 
                                            device, 
                                            img_feats_local_inuse, 
                                            pc_feats_local_inuse, 
                                            self.cfgs)

                local_positive_mask = torch.gt(overlap_ratio_matrix_inuse, self.cfgs.local_positive_overlap_ratio_threshold)
                local_negative_mask = torch.lt(overlap_ratio_matrix_inuse, self.cfgs.local_negative_overlap_ratio_threshold)

                local_positive_mask[~in_batch_flag_matrix] = False
                local_negative_mask[~in_batch_flag_matrix] = False

                local_positive_mask = local_positive_mask.permute(0, 2, 1)
                local_negative_mask = local_negative_mask.permute(0, 2, 1)

                data_output['img_feats_local'] = img_pair_embeddings
                data_output['pc_feats_local'] = pc_pair_embeddings
                data_output['local_positive_mask'] = local_positive_mask
                data_output['local_negative_mask'] = local_negative_mask

                if 'correspndence_among_pc' in self.cfgs.keys() and self.cfgs.correspndence_among_pc:
                    (onlypc_overlap_ratio_matrix, 
                     onlypc_embeddings1, 
                     onlypc_embeddings2, 
                     onlypc_in_batch_flag_matrix) = generate_single_correspondence_for_pc(f_points,
                                                                                          data_dict,
                                                                                          device,
                                                                                          pc_feats_local_inuse,
                                                                                          self.cfgs)
                    onlypc_local_positive_mask = torch.gt(onlypc_overlap_ratio_matrix, self.cfgs.onlypc_local_positive_overlap_ratio_threshold)
                    onlypc_local_negative_mask = torch.lt(onlypc_overlap_ratio_matrix, self.cfgs.onlypc_local_negative_overlap_ratio_threshold)

                    onlypc_local_positive_mask[~onlypc_in_batch_flag_matrix] = False
                    onlypc_local_negative_mask[~onlypc_in_batch_flag_matrix] = False

                    data_output['onlypc_embeddings1'] = onlypc_embeddings1
                    data_output['onlypc_embeddings2'] = onlypc_embeddings2
                    data_output['onlypc_local_positive_mask'] = onlypc_local_positive_mask
                    data_output['onlypc_local_negative_mask'] = onlypc_local_negative_mask

            img_feat_inuse = self.phase22_aggregator(img_feats_aggr_inuse)
            pc_feat_inuse = self.phase22_aggregator(pc_feats_aggr_inuse)
            data_output['embeddings2'] = pc_feat_inuse
            data_output['embeddings1'] = img_feat_inuse

        elif self.cfgs.phase == 23:
            # use the features directly
            img_feat_inuse = self.phase23_aggregator(img_feats_inuse)
            pc_feat_inuse = self.phase23_aggregator(pc_feats_inuse)
            data_output['embeddings2'] = pc_feat_inuse
            data_output['embeddings1'] = img_feat_inuse
            # compute key features
            with torch.no_grad():
                for param_q, param_k in zip(
                    self.pc_encoder.parameters(), self.momentum_pc_encoder.parameters()
                ):
                    param_k.data = param_k.data * self.cfgs.phase23_m + param_q.data * (1.0 - self.cfgs.phase23_m)
                for param_q, param_k in zip(
                    self.image_encoder.parameters(), self.momentum_image_encoder.parameters()
                ):
                    param_k.data = param_k.data * self.cfgs.phase23_m + param_q.data * (1.0 - self.cfgs.phase23_m)
                for param_q, param_k in zip(
                    self.phase23_aggregator.parameters(), self.momentum_phase23_aggregator.parameters()
                ):
                    param_k.data = param_k.data * self.cfgs.phase23_m + param_q.data * (1.0 - self.cfgs.phase23_m)
                c_img_feats_momentum, f_img_feats_momentum = self.momentum_image_encoder(data_dict['images'])
                (
                    f_pc_feats_momentum, 
                    c_pc_feats_momentum, 
                    f_points_momentum, 
                    c_points_momentum, 
                    f_masks_momentum, 
                    c_masks_momentum
                ) = self.momentum_pc_encoder(data_dict['clouds'])
                if self.cfgs.image_out_layer == 1:
                    momentum_img_feats_inuse = c_img_feats_momentum
                elif self.cfgs.image_out_layer == 2:
                    momentum_img_feats_inuse = f_img_feats_momentum
                if self.cfgs.pc_out_layer == 1:
                    momentum_pc_feats_inuse = c_pc_feats_momentum
                elif self.cfgs.pc_out_layer == 2:
                    momentum_pc_feats_inuse = f_pc_feats_momentum
                img_feat_inuse_momentum = self.momentum_phase23_aggregator(momentum_img_feats_inuse)
                pc_feat_inuse_momentum = self.momentum_phase23_aggregator(momentum_pc_feats_inuse)
                data_output['key_embeddings2'] = pc_feat_inuse_momentum
                data_output['key_embeddings1'] = img_feat_inuse_momentum


        elif self.cfgs.phase == 10:
            pass
            # img_h_f = f_img_feats[2]
            # img_w_f = f_img_feats[3]
            # img_h_c = c_img_feats[2]
            # img_w_c = c_img_feats[3]

            # # generate 3D patches
            # (
            #     _, 
            #     pcd_node_sizes, 
            #     pcd_node_masks, 
            #     pcd_node_knn_indices, 
            #     pcd_node_knn_masks
            #     ) = point_to_node_partition_batch(
            #     f_points,
            #     c_points,
            #     f_masks,
            #     c_masks,
            #     point_limit=self.cfgs.pcnet_point_limit,
            #     gather_points=True,
            #     return_count=True,
            # )
            # # generate img_pixels
            # img_pixels = create_meshgrid(img_h_f, img_w_f).float()  # (img_h_f, img_w_f, 2)
            # img_pixels = img_pixels.reshape(-1, 2)  # (img_h_f * img_w_f, 2)

            # # generate 2D patches
            # (
            #     img_node_knn_pixels,
            #     img_node_knn_indices
            # ) = patchify_CFF(
            #     img_pixels, 
            #     img_h_f, 
            #     img_w_f, 
            #     img_h_c, 
            #     img_w_c, 
            #     stride=1
            #     ) # (img_h_c, img_w_c, img_h_f/img_h_c/stride * img_w_f/img_w_c/stride), (img_h_c, img_w_c, img_h_f/img_h_c/stride * img_w_f/img_w_c/stride, 2)
            # # generate pcd_node_masks
            # pcd_node_masks = torch.logical_and(pcd_node_masks, torch.gt(pcd_node_sizes, self.cfgs.pcd_min_node_size))
            # # pad the points
            # pcd_padded_points_f = torch.cat([f_points, (torch.ones_like(f_points[:, 0, :]) * 1e10).unsqueeze(1)], dim=1) # (B, N_f + 1, 3)
            # # generate pcd pixels
            # pcd_pixels_f = render(f_points, data_dict['intrinsics'], extrinsics=data_dict['transform'], rounding=False) # (B, N_f, 2)
            # # generate pcd_knn_points
            # pcd_padded_points_flattenned = pcd_padded_points_f.reshape(-1, 3)
            # pcd_node_knn_indices_add = (torch.arange(0, pcd_node_knn_indices.shape[0], device=f_points.device, dtype=torch.long) * pcd_padded_points_f.shape[1]).unsqueeze(1).unsqueeze(1).expand_as(pcd_node_knn_indices) # (B, M, K)
            # pcd_node_knn_points = index_select(pcd_padded_points_flattenned, pcd_node_knn_indices_add, dim=0) # (B, M, K, 3)
            # # generate pcd_knn_pixels
            # pcd_padded_pixels_f = torch.cat([pcd_pixels_f, (torch.ones_like(pcd_pixels_f[:, 0, :]) * 1e10).unsqueeze(1)], dim=1) # (B, N_f + 1, 2)
            # pcd_padded_pixels_flattenned = pcd_padded_pixels_f.reshape(-1, 2)
            # pcd_node_knn_pixels = index_select(pcd_padded_pixels_flattenned, pcd_node_knn_indices_add, dim=0) # (B, M, K, 2)
            # # get 2D-3D truth correspondences
            # img_masks = (torch.ones_like(img_node_knn_indices[:, 0], dtype=torch.bool)).unsqueeze(0).expand(pcd_node_masks.shape[0], -1) # (B, M)
            # img_node_knn_pixels = img_node_knn_pixels.unsqueeze(0).expand(pcd_node_masks.shape[0], -1, -1, -1) # (B, M, K, 2)
            # image_knn_masks = (torch.ones_like(img_node_knn_indices)).unsqueeze(0).expand(pcd_node_masks.shape[0], -1, -1) # (B, M, K)
            # (
            #     gt_batch_corr_indices,
            #     gt_img_node_corr_indices,
            #     gt_pcd_node_corr_indices,
            #     gt_img_node_corr_overlaps,
            #     gt_pcd_node_corr_overlaps,
            # ) = get_2d3d_node_correspondences_batch(
            #     img_masks,
            #     img_node_knn_pixels,
            #     image_knn_masks, 
            #     pcd_node_masks, 
            #     pcd_node_knn_pixels, 
            #     pcd_node_knn_masks, 
            #     self.cfgs.matching_radius_2d
            #     )
        return data_output

class CMVPRv2(nn.Module):

    def __init__(self, config, out_dim):
        super(CMVPRv2, self).__init__()
        self.image_encoder = ImageEncoder(config.image_encoder_type, config.image_encoder_cfgs, out_dim, config.image_encoder_out_layer)
        self.render_encoder = ImageEncoder(config.render_encoder_type, config.render_encoder_cfgs, out_dim, config.render_encoder_out_layer)
        self.cfgs = config
        self.render_aggregator = aggregator(self.cfgs.render_aggregator_type, self.cfgs.render_aggregator_cfgs, out_dim)
        self.image_aggregator = aggregator(self.cfgs.image_aggregator_type, self.cfgs.image_aggregator_cfgs, out_dim)
        if self.cfgs.phase == 1:
            if self.cfgs.two_aggr:
                pass
            else:
                self.phase1_aggregator = aggregator(self.cfgs.phase1_aggregator_type, self.cfgs.phase1_aggregator_cfgs, out_dim)
        if self.cfgs.phase == 7 or self.cfgs.phase == 2: # created for ModalLink
            self.phase7_mlp = nn.Sequential(nn.Conv2d(out_dim, 128, kernel_size=(1, 1)),
                        nn.ReLU(),
                        nn.Conv2d(128, self.cfgs.phase7_K, kernel_size=(1, 1)),
                        nn.ReLU())
            self.phase7_aggregator_mlp = aggregator(self.cfgs.phase7_aggregator_mlp_type, self.cfgs.phase7_aggregator_mlp_cfgs, self.cfgs.phase7_K)
            self.phase7_aggregator_NMF = aggregator(self.cfgs.phase7_aggregator_NMF_type, self.cfgs.phase7_aggregator_NMF_cfgs, self.cfgs.phase7_K)
            self.phase7_aggregator_normal = aggregator(self.cfgs.phase7_aggregator_normal_type, self.cfgs.phase7_aggregator_normal_cfgs, out_dim)
            self.phase7_fuse = Proj(embeddim=(out_dim + 2 * self.cfgs.phase7_K),
                                    fuse_type=None,
                                    proj_type=1)
            self.phase7_relu = nn.ReLU(inplace=True)
        
        if self.cfgs.phase == 3:
            if 'two_aggr' in self.cfgs.keys() and self.cfgs.two_aggr:
                pass
            else:
                self.phase3_aggregator = aggregator(self.cfgs.phase3_aggregator_type, self.cfgs.phase3_aggregator_cfgs, out_dim)
        
        if self.cfgs.phase == 4:
            self.pc_bev_encoder = ImageEncoder(config.pc_bev_encoder_type, config.pc_bev_encoder_cfgs, out_dim, config.pc_bev_encoder_out_layer)
            self.image_bev_encoder = ImageEncoder(config.image_bev_encoder_type, config.image_bev_encoder_cfgs, out_dim, config.image_bev_encoder_out_layer)
            if config.pc_bev_encoder_type == 'ResUNetmmseg':
                for param in self.pc_bev_encoder.module.decoder_head_1.parameters():
                    param.requires_grad = False
            elif config.pc_bev_encoder_type == 'ResFPNmmseg':
                for param in self.pc_bev_encoder.module.neck.parameters():
                    param.requires_grad = False
                for param in self.pc_bev_encoder.module.decode_head.parameters():
                    param.requires_grad = False
            if config.image_bev_encoder_type == 'ResUNetmmseg':
                for param in self.image_bev_encoder.module.decoder_head_1.parameters():
                    param.requires_grad = False
            elif config.image_bev_encoder_type == 'ResFPNmmseg':
                for param in self.image_bev_encoder.module.neck.parameters():
                    param.requires_grad = False
                for param in self.image_bev_encoder.module.decode_head.parameters():
                    param.requires_grad = False
            if not self.cfgs.two_aggr:
                self.phase4_aggregator = aggregator(self.cfgs.phase4_aggregator_type, self.cfgs.phase4_aggregator_cfgs, out_dim)
        
        if self.cfgs.phase == 5:
            self.pc_bev_encoder = ImageEncoder(config.pc_bev_encoder_type, config.pc_bev_encoder_cfgs, out_dim, config.pc_bev_encoder_out_layer)
            self.image_bev_encoder = ImageEncoder(config.image_bev_encoder_type, config.image_bev_encoder_cfgs, out_dim, config.image_bev_encoder_out_layer)
            if config.pc_bev_encoder_type == 'ResUNetmmseg':
                for param in self.pc_bev_encoder.module.decoder_head_1.parameters():
                    param.requires_grad = False
            elif config.pc_bev_encoder_type == 'ResFPNmmseg':
                for param in self.pc_bev_encoder.module.neck.parameters():
                    param.requires_grad = False
                for param in self.pc_bev_encoder.module.decode_head.parameters():
                    param.requires_grad = False
            if config.image_bev_encoder_type == 'ResUNetmmseg':
                for param in self.image_bev_encoder.module.decoder_head_1.parameters():
                    param.requires_grad = False
            elif config.image_bev_encoder_type == 'ResFPNmmseg':
                for param in self.image_bev_encoder.module.neck.parameters():
                    param.requires_grad = False
                for param in self.image_bev_encoder.module.decode_head.parameters():
                    param.requires_grad = False
            
            if not self.cfgs.two_aggr:
                self.phase5_aggregator1 = aggregator(self.cfgs.phase5_aggregator1_type, self.cfgs.phase5_aggregator1_cfgs, out_dim)
                self.phase5_aggregator2 = aggregator(self.cfgs.phase5_aggregator2_type, self.cfgs.phase5_aggregator2_cfgs, out_dim)
            else:
                self.phase5_aggregator1 = aggregator(self.cfgs.phase5_aggregator1_type, self.cfgs.phase5_aggregator1_cfgs, out_dim)
                self.phase5_aggregator2 = aggregator(self.cfgs.phase5_aggregator2_type, self.cfgs.phase5_aggregator2_cfgs, out_dim)
                self.phase5_aggregator3 = aggregator(self.cfgs.phase5_aggregator3_type, self.cfgs.phase5_aggregator3_cfgs, out_dim)
                self.phase5_aggregator4 = aggregator(self.cfgs.phase5_aggregator4_type, self.cfgs.phase5_aggregator4_cfgs, out_dim)
            if self.cfgs.phase5_aggregator_feature_output_type == 'fc':
                self.phase5_fc1 = Proj(embeddim=out_dim, 
                                       fuse_type='concat', 
                                       proj_type=self.cfgs.phase5_proj_type1)
                self.phase5_fc2 = Proj(embeddim=out_dim,
                                        fuse_type='concat',
                                        proj_type=self.cfgs.phase5_proj_type2)
            if self.cfgs.phase5_attention:
                phase5_attention_block = ResidualAttention(num_layers=self.cfgs.phase5_attention_num_layers,
                                                                 d_model=out_dim,
                                                                 n_head=self.cfgs.phase5_attention_n_head,
                                                                 att_type='cross',
                                                                 out_norm=LayerNorm if self.cfgs.phase5_attention_out_norm else None)
                self.phase5_attention_block1 = copy.deepcopy(phase5_attention_block)
                self.phase5_attention_block2 = copy.deepcopy(phase5_attention_block)
                if 'attn_pos' in self.cfgs.keys() and self.cfgs.attn_pos == 'after':
                    self.phase5_aggregator_attention = aggregator(self.cfgs.phase5_aggregator_attention_type, self.cfgs.phase5_aggregator_attention_cfgs, out_dim)
            
            if 'phase5_attention_map' in self.cfgs.keys() and self.cfgs.phase5_attention_map:
                if self.cfgs.phase5_fc_attn_feat_type == 1:
                    self.fc_attn_feat1 = nn.Sequential(nn.Linear(2 * out_dim, out_dim // (2 * 1)), 
                                                    nn.BatchNorm1d(out_dim // (2 * 1)),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(out_dim // (2 * 1), out_dim // (2 * 3)),
                                                    nn.BatchNorm1d(out_dim // (2 * 3)),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(out_dim // (2 * 3), 2),
                                                    nn.BatchNorm1d(2),
                                                    nn.Sigmoid())
                    self.fc_attn_feat2 = nn.Sequential(nn.Linear(2 * out_dim, out_dim // (2 * 1)), 
                                                    nn.BatchNorm1d(out_dim // (2 * 1)),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(out_dim // (2 * 1), out_dim // (2 * 3)),
                                                    nn.BatchNorm1d(out_dim // (2 * 3)),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(out_dim // (2 * 3), 2),
                                                    nn.BatchNorm1d(2),
                                                    nn.Sigmoid())
                elif self.cfgs.phase5_fc_attn_feat_type == 2:
                    self.fc_attn_feat1 = nn.Sequential(nn.Linear(2 * out_dim, out_dim // (2 * 1)), 
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(out_dim // (2 * 1), out_dim // (2 * 3)),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(out_dim // (2 * 3), 2),
                                                    nn.Sigmoid())
                    self.fc_attn_feat2 = nn.Sequential(nn.Linear(2 * out_dim, out_dim // (2 * 1)), 
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(out_dim // (2 * 1), out_dim // (2 * 3)),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(out_dim // (2 * 3), 2),
                                                    nn.Sigmoid())
                else:
                    raise NotImplementedError
                self.phase5_aggregator1_attention_map = aggregator(self.cfgs.phase5_aggregator_attention_map_type, 
                                                                  self.cfgs.phase5_aggregator_attention_map_cfgs, 
                                                                  out_dim)
                self.phase5_aggregator2_attention_map = aggregator(self.cfgs.phase5_aggregator_attention_map_type, 
                                                                  self.cfgs.phase5_aggregator_attention_map_cfgs, 
                                                                  out_dim)
            
        if self.cfgs.phase == 6:
            self.phase6_aggregator = aggregator(self.cfgs.phase6_aggregator_type, self.cfgs.phase6_aggregator_cfgs, out_dim)
            self.momentum_image_encoder = ImageEncoder(config.image_encoder_type, config.image_encoder_cfgs, out_dim, config.image_encoder_out_layer)
            self.momentum_render_encoder = ImageEncoder(config.render_encoder_type, config.render_encoder_cfgs, out_dim, config.render_encoder_out_layer)
            self.momentum_phase6_aggregator = aggregator(self.cfgs.phase6_aggregator_type, self.cfgs.phase6_aggregator_cfgs, out_dim)
            for param_q, param_k in zip(
                self.render_encoder.parameters(), self.momentum_render_encoder.parameters()
            ):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
            for param_q, param_k in zip(
                self.image_encoder.parameters(), self.momentum_image_encoder.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(
                self.phase6_aggregator.parameters(), self.momentum_phase6_aggregator.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
        
        self.out_dim = out_dim
    
    def forward(self, data_dict):
        data_output = {}
        if (self.cfgs.phase == 1 
            or self.cfgs.phase == 2 
            or self.cfgs.phase == 3
            or self.cfgs.phase == 6 
            or self.cfgs.phase == 7):
            if 'multi_layer' in self.cfgs.keys() and self.cfgs.multi_layer:
                img_1_feats, img_2_feats, img_3_feats = self.image_encoder(data_dict['images'])
                render_1_feats, render_2_feats, render_3_feats = self.render_encoder(data_dict['render_imgs'])
            else:
                c_img_feats, f_img_feats = self.image_encoder(data_dict['images'])
                c_render_feats, f_render_feats = self.render_encoder(data_dict['render_imgs'])
        elif self.cfgs.phase == 4:
            c_img_bev_feats, f_img_bev_feats = self.image_bev_encoder(data_dict['image_bevs'])
            c_pc_bev_feats, f_pc_bev_feats = self.pc_bev_encoder(data_dict['pc_bevs'])
        elif self.cfgs.phase == 5:
            c_img_bev_feats, f_img_bev_feats = self.image_bev_encoder(data_dict['image_bevs'])
            c_pc_bev_feats, f_pc_bev_feats = self.pc_bev_encoder(data_dict['pc_bevs'])
            c_img_feats, f_img_feats = self.image_encoder(data_dict['images'])
            c_render_feats, f_render_feats = self.render_encoder(data_dict['render_imgs'])
        if self.cfgs.image_out_layer == 1:
            img_feats_inuse = c_img_feats
        elif self.cfgs.image_out_layer == 2:
            img_feats_inuse = f_img_feats
        else:
            img_feats_inuse = None
        if self.cfgs.render_out_layer == 1:
            render_feats_inuse = c_render_feats
        elif self.cfgs.render_out_layer == 2:
            render_feats_inuse = f_render_feats
        else:
            render_feats_inuse = None
        if self.cfgs.phase == 1:
            # use the features directly
            if self.cfgs.two_aggr:
                render_aggr_feat = self.render_aggregator(render_feats_inuse)
                img_aggr_feat = self.image_aggregator(img_feats_inuse)
            else:
                render_aggr_feat = self.phase1_aggregator(render_feats_inuse)
                img_aggr_feat = self.phase1_aggregator(img_feats_inuse)
            data_output['embeddings2'] = render_aggr_feat
            data_output['embeddings1'] = img_aggr_feat
        elif self.cfgs.phase == 2:
            feats_inuse = torch.cat([render_feats_inuse, img_feats_inuse], dim=0)
            feats_inuse = F.normalize(feats_inuse, p=2, dim=1)
            feats_after_mlp = self.phase7_mlp(feats_inuse)
            feats_after_mlp = F.normalize(feats_after_mlp, p=2, dim=1)
            aggr_mlp_feat = self.phase7_aggregator_mlp(feats_after_mlp)
            aggr_mlp_feat = F.normalize(aggr_mlp_feat, p=2, dim=1)

            with torch.no_grad(): # nmf is not derivable
                features = feats_inuse.contiguous()
                b, h, w = features.size(0), features.size(2), features.size(3)
                features = self.phase7_relu(features)
                flat_features = features.permute(0, 2, 3, 1).contiguous().view(-1, features.size(1))
                W, _ = NMF(flat_features, self.cfgs.phase7_K, random_seed=1, cuda=True, max_iter=self.cfgs.phase7_max_iter, verbose=False)
                isnan = torch.sum(torch.isnan(W).float())
                while isnan > 0:
                    print('nan detected. trying to resolve the nmf.')
                    W, _ = NMF(flat_features, self.cfgs.phase7_K, random_seed=random.randint(0, 255), cuda=True, max_iter=self.cfgs.phase7_max_iter, verbose=False)
                    isnan = torch.sum(torch.isnan(W).float())
                heatmaps = W.view(b, h, w, self.cfgs.phase7_K).permute(0,3,1,2)
                heatmaps = F.normalize(heatmaps, p=2, dim=1)
                heatmaps.requires_grad = False
                aggr_feat_NMF = self.phase7_aggregator_NMF(heatmaps)
                aggr_feat_NMF = F.normalize(aggr_feat_NMF, p=2, dim=1)

            aggr_feat_normal = self.phase7_aggregator_normal(feats_inuse)
            aggr_feat_normal = F.normalize(aggr_feat_normal, p=2, dim=1)
            aggr_feat_to_fuse = torch.cat([aggr_feat_normal, aggr_feat_NMF, aggr_mlp_feat], dim=1)
            aggr_fused_feat = self.phase7_fuse(aggr_feat_to_fuse)
            aggr_fused_feat = aggr_fused_feat.type(torch.float32)
            feature_length = b // 2
            img_aggr_fused_feat = aggr_fused_feat[:feature_length, :]
            render_aggr_fused_feat = aggr_fused_feat[feature_length:, :]
            data_output['embeddings1'] = img_aggr_fused_feat
            data_output['embeddings2'] = render_aggr_fused_feat
        elif self.cfgs.phase == 3:
            if self.cfgs.phase3_render_aggr_layer == 1:
                render_feats_aggr_inuse = c_render_feats
            elif self.cfgs.phase3_render_aggr_layer == 2:
                render_feats_aggr_inuse = f_render_feats
            if self.cfgs.phase3_image_aggr_layer == 1:
                img_feats_aggr_inuse = c_img_feats
            elif self.cfgs.phase3_image_aggr_layer == 2:
                img_feats_aggr_inuse = f_img_feats
            if self.cfgs.phase3_render_local_layer == 1:
                render_feats_local_inuse = c_render_feats
            elif self.cfgs.phase3_render_local_layer == 2:
                render_feats_local_inuse = f_render_feats
            if self.cfgs.phase3_image_local_layer == 1:
                img_feats_local_inuse = c_img_feats
            elif self.cfgs.phase3_image_local_layer == 2:
                img_feats_local_inuse = f_img_feats

            if 'two_aggr' in self.cfgs.keys() and self.cfgs.two_aggr:
                render_aggr_feat = self.render_aggregator(render_feats_aggr_inuse)
                img_aggr_feat = self.image_aggregator(img_feats_aggr_inuse)
            else:
                render_aggr_feat = self.phase3_aggregator(render_feats_aggr_inuse)
                img_aggr_feat = self.phase3_aggregator(img_feats_aggr_inuse)

            data_output['embeddings2'] = render_aggr_feat
            data_output['embeddings1'] = img_aggr_feat
            if self.training:
                pixels_selected_indices = data_dict['pixels_selected_indices']
                render_pixels_selected_indices = data_dict['render_pixels_selected_indices']
                local_positive_mask = data_dict['local_positive_mask'].permute(0, 2, 1)
                local_negative_mask = data_dict['local_negative_mask'].permute(0, 2, 1)
                pixels_selected_indices = pixels_selected_indices[:, :, 1] * img_feats_local_inuse.shape[3] + pixels_selected_indices[:, :, 0]
                img_feats_local = torch.gather(img_feats_local_inuse.flatten(2).permute(0, 2, 1),
                                                dim=1,
                                                index=pixels_selected_indices.unsqueeze(-1).expand(-1, -1, self.out_dim))
                render_feats_local = torch.gather(render_feats_local_inuse.flatten(2).permute(0, 2, 1),
                                                  dim=1,
                                                  index=render_pixels_selected_indices.unsqueeze(-1).expand(-1, -1, self.out_dim))
                data_output['img_feats_local'] = img_feats_local
                data_output['pc_feats_local'] = render_feats_local
                data_output['local_positive_mask'] = local_positive_mask
                data_output['local_negative_mask'] = local_negative_mask
        elif self.cfgs.phase == 4:
            if self.cfgs.phase4_pc_bev_aggr_layer == 1:
                pc_bev_feats_aggr_inuse = c_pc_bev_feats
            elif self.cfgs.phase4_pc_bev_aggr_layer == 2:
                pc_bev_feats_aggr_inuse = f_pc_bev_feats
            if self.cfgs.phase4_image_bev_aggr_layer == 1:
                image_bev_feats_aggr_inuse = c_img_bev_feats
            elif self.cfgs.phase4_image_bev_aggr_layer == 2:
                image_bev_feats_aggr_inuse = f_img_bev_feats
            

            if 'two_aggr' in self.cfgs.keys() and self.cfgs.two_aggr:
                pc_bev_aggr_feat = self.render_aggregator(pc_bev_feats_aggr_inuse)
                img_bev_aggr_feat = self.image_aggregator(image_bev_feats_aggr_inuse)
            else:
                pc_bev_aggr_feat = self.phase4_aggregator(pc_bev_feats_aggr_inuse)
                img_bev_aggr_feat = self.phase4_aggregator(image_bev_feats_aggr_inuse)

            data_output['embeddings2'] = pc_bev_aggr_feat
            data_output['embeddings1'] = img_bev_aggr_feat
        
        elif self.cfgs.phase == 5:
            if self.cfgs.phase5_pc_bev_aggr_layer == 1:
                pc_bev_feats_aggr_inuse = c_pc_bev_feats
            elif self.cfgs.phase5_pc_bev_aggr_layer == 2:
                pc_bev_feats_aggr_inuse = f_pc_bev_feats
            if self.cfgs.phase5_image_bev_aggr_layer == 1:
                image_bev_feats_aggr_inuse = c_img_bev_feats
            elif self.cfgs.phase5_image_bev_aggr_layer == 2:
                image_bev_feats_aggr_inuse = f_img_bev_feats
            
            if self.cfgs.phase5_render_aggr_layer == 1:
                render_feats_aggr_inuse = c_render_feats
            elif self.cfgs.phase5_render_aggr_layer == 2:
                render_feats_aggr_inuse = f_render_feats
            if self.cfgs.phase5_image_aggr_layer == 1:
                img_feats_aggr_inuse = c_img_feats
            elif self.cfgs.phase5_image_aggr_layer == 2:
                img_feats_aggr_inuse = f_img_feats
            
            if self.cfgs.phase5_attention and (('attn_pos' not in self.cfgs.keys()) or self.cfgs.attn_pos == 'before'):
                image_bev_feats_aggr_inuse, img_feats_aggr_inuse = self.phase5_attention_block1(image_bev_feats_aggr_inuse.flatten(2).permute(0, 2, 1), img_feats_aggr_inuse.flatten(2).permute(0, 2, 1))
                pc_bev_feats_aggr_inuse, render_feats_aggr_inuse = self.phase5_attention_block2(pc_bev_feats_aggr_inuse.flatten(2).permute(0, 2, 1), render_feats_aggr_inuse.flatten(2).permute(0, 2, 1))
                image_bev_feats_aggr_inuse = image_bev_feats_aggr_inuse.permute(0, 2, 1)
                img_feats_aggr_inuse = img_feats_aggr_inuse.permute(0, 2, 1)
                pc_bev_feats_aggr_inuse = pc_bev_feats_aggr_inuse.permute(0, 2, 1)
                render_feats_aggr_inuse = render_feats_aggr_inuse.permute(0, 2, 1)

            if 'two_aggr' in self.cfgs.keys() and self.cfgs.two_aggr:
                pc_bev_aggr_feat = self.phase5_aggregator1(pc_bev_feats_aggr_inuse)
                img_bev_aggr_feat = self.phase5_aggregator2(image_bev_feats_aggr_inuse)
                render_aggr_feat = self.phase5_aggregator3(render_feats_aggr_inuse)
                img_aggr_feat = self.phase5_aggregator4(img_feats_aggr_inuse)
            else:
                pc_bev_aggr_feat = self.phase5_aggregator1(pc_bev_feats_aggr_inuse)
                img_bev_aggr_feat = self.phase5_aggregator1(image_bev_feats_aggr_inuse)
                render_aggr_feat = self.phase5_aggregator2(render_feats_aggr_inuse)
                img_aggr_feat = self.phase5_aggregator2(img_feats_aggr_inuse)
            
            if self.cfgs.phase5_attention and 'attn_pos' in self.cfgs.keys() and self.cfgs.attn_pos == 'after':
                if self.cfgs.phase5_attention_after_num == 1:
                    pc_bev_aggr_feat, render_aggr_feat = self.phase5_attention_block1(pc_bev_aggr_feat.unsqueeze(1), render_aggr_feat.unsqueeze(1))
                    img_bev_aggr_feat, img_aggr_feat = self.phase5_attention_block1(img_bev_aggr_feat.unsqueeze(1), img_aggr_feat.unsqueeze(1))
                elif self.cfgs.phase5_attention_after_num == 2:
                    pc_bev_aggr_feat, render_aggr_feat = self.phase5_attention_block1(pc_bev_aggr_feat.unsqueeze(1), render_aggr_feat.unsqueeze(1))
                    img_bev_aggr_feat, img_aggr_feat = self.phase5_attention_block2(img_bev_aggr_feat.unsqueeze(1), img_aggr_feat.unsqueeze(1))
                else:
                    raise NotImplementedError
                pc_bev_aggr_feat = pc_bev_aggr_feat.squeeze(1)
                render_aggr_feat = render_aggr_feat.squeeze(1)
                img_bev_aggr_feat = img_bev_aggr_feat.squeeze(1)
                img_aggr_feat = img_aggr_feat.squeeze(1)
                pc_aggr_feat = self.phase5_aggregator_attention(torch.stack([pc_bev_aggr_feat, render_aggr_feat], dim=-1))
                img_aggr_feat = self.phase5_aggregator_attention(torch.stack([img_bev_aggr_feat, img_aggr_feat], dim=-1))
                data_output['embeddings2'] = pc_aggr_feat
                data_output['embeddings1'] = img_aggr_feat
                return data_output

            if 'phase5_attention_map' in self.cfgs.keys() and self.cfgs.phase5_attention_map:
                pc_bev_attn_feat = f_pc_bev_feats.flatten(2, 3)
                img_bev_attn_feat = f_img_bev_feats.flatten(2, 3)
                render_attn_feat = f_render_feats.flatten(2, 3)
                img_attn_feat = f_img_feats.flatten(2, 3)
                pc_bev_attn_feat = F.softmax(pc_bev_attn_feat, dim=-1)
                img_bev_attn_feat = F.softmax(img_bev_attn_feat, dim=-1)
                render_attn_feat = F.softmax(render_attn_feat, dim=-1)
                img_attn_feat = F.softmax(img_attn_feat, dim=-1)
                if self.cfgs.phase5_two_aggregator_attention:
                    pc_bev_attn_feat = self.phase5_aggregator1_attention_map(pc_bev_attn_feat)
                    img_bev_attn_feat = self.phase5_aggregator2_attention_map(img_bev_attn_feat)
                    render_attn_feat = self.phase5_aggregator1_attention_map(render_attn_feat)
                    img_attn_feat = self.phase5_aggregator2_attention_map(img_attn_feat)
                    pc_weight_vector = self.fc_attn_feat1(torch.cat([pc_bev_attn_feat, render_attn_feat], dim=-1))
                    img_weight_vector = self.fc_attn_feat2(torch.cat([img_bev_attn_feat, img_attn_feat], dim=-1))
                else:
                    pc_bev_attn_feat = self.phase5_aggregator1_attention_map(pc_bev_attn_feat)
                    img_bev_attn_feat = self.phase5_aggregator1_attention_map(img_bev_attn_feat)
                    render_attn_feat = self.phase5_aggregator1_attention_map(render_attn_feat)
                    img_attn_feat = self.phase5_aggregator1_attention_map(img_attn_feat)
                    pc_weight_vector = self.fc_attn_feat1(torch.cat([pc_bev_attn_feat, render_attn_feat], dim=-1))
                    img_weight_vector = self.fc_attn_feat1(torch.cat([img_bev_attn_feat, img_attn_feat], dim=-1))
                img_aggr_feat_combined = torch.cat([img_weight_vector[:, 0:1] * img_bev_aggr_feat, img_weight_vector[:, 1:] * img_aggr_feat], dim=-1)
                pc_aggr_feat_combined = torch.cat([pc_weight_vector[:, 0:1] * pc_bev_aggr_feat, pc_weight_vector[:, 1:] * render_aggr_feat], dim=-1)
                data_output['embeddings2'] = pc_aggr_feat_combined
                data_output['embeddings1'] = img_aggr_feat_combined
                if 'phase5_attention_map_rerank' in self.cfgs.keys() and self.cfgs.phase5_attention_map_rerank:
                    data_output['embeddings4'] = pc_bev_aggr_feat
                    data_output['embeddings3'] = img_bev_aggr_feat
                    data_output['embeddings6'] = render_aggr_feat
                    data_output['embeddings5'] = img_aggr_feat
                return data_output
            
            if self.cfgs.phase5_aggregator_feature_output_type == 'cat':
                img_aggr_feat = torch.cat([img_bev_aggr_feat, img_aggr_feat], dim=-1)
                pc_aggr_feat = torch.cat([pc_bev_aggr_feat, render_aggr_feat], dim=-1)
                data_output['embeddings2'] = pc_aggr_feat
                data_output['embeddings1'] = img_aggr_feat
            elif self.cfgs.phase5_aggregator_feature_output_type == 'fc':
                img_aggr_feat = self.phase5_fc1(torch.cat([img_bev_aggr_feat, img_aggr_feat], dim=-1))
                pc_aggr_feat = self.phase5_fc2(torch.cat([pc_bev_aggr_feat, render_aggr_feat], dim=-1))
                data_output['embeddings2'] = pc_aggr_feat
                data_output['embeddings1'] = img_aggr_feat
            elif self.cfgs.phase5_aggregator_feature_output_type == 'cat_normalize':
                img_bev_aggr_feat = F.normalize(img_bev_aggr_feat, p=2, dim=-1)
                img_aggr_feat = F.normalize(img_aggr_feat, p=2, dim=-1)
                pc_bev_aggr_feat = F.normalize(pc_bev_aggr_feat, p=2, dim=-1)
                render_aggr_feat = F.normalize(render_aggr_feat, p=2, dim=-1)
                img_aggr_feat = torch.cat([img_bev_aggr_feat, img_aggr_feat], dim=-1)
                pc_aggr_feat = torch.cat([pc_bev_aggr_feat, render_aggr_feat], dim=-1)
                data_output['embeddings2'] = pc_aggr_feat
                data_output['embeddings1'] = img_aggr_feat
            elif self.cfgs.phase5_aggregator_feature_output_type == 'split':
                data_output['embeddings2'] = pc_bev_aggr_feat
                data_output['embeddings1'] = img_bev_aggr_feat
                data_output['embeddings4'] = render_aggr_feat
                data_output['embeddings3'] = img_aggr_feat
            else:
                raise NotImplementedError
        elif self.cfgs.phase == 6:
            # use the features directly
            render_aggr_feat = self.phase6_aggregator(render_feats_inuse)
            img_aggr_feat = self.phase6_aggregator(img_feats_inuse)
            data_output['embeddings2'] = render_aggr_feat
            data_output['embeddings1'] = img_aggr_feat
            with torch.no_grad():
                for param_q, param_k in zip(                
                    self.render_encoder.parameters(), self.momentum_render_encoder.parameters()
                ):
                    param_k.data = param_k.data * self.cfgs.phase6_m + param_q.data * (1.0 - self.cfgs.phase6_m)
                for param_q, param_k in zip(
                    self.image_encoder.parameters(), self.momentum_image_encoder.parameters()
                ):
                    param_k.data = param_k.data * self.cfgs.phase6_m + param_q.data * (1.0 - self.cfgs.phase6_m)
                for param_q, param_k in zip(
                    self.phase6_aggregator.parameters(), self.momentum_phase6_aggregator.parameters()
                ):
                    param_k.data = param_k.data * self.cfgs.phase6_m + param_q.data * (1.0 - self.cfgs.phase6_m)
                c_img_feats_momentum, f_img_feats_momentum = self.momentum_image_encoder(data_dict['images'])
                c_render_feats_momentum, f_render_feats_momentum = self.momentum_render_encoder(data_dict['render_imgs'])
                if self.cfgs.image_out_layer == 1:
                    img_feats_inuse_momentum = c_img_feats_momentum
                elif self.cfgs.image_out_layer == 2:
                    img_feats_inuse_momentum = f_img_feats_momentum
                if self.cfgs.render_out_layer == 1:
                    render_feats_inuse_momentum = c_render_feats_momentum
                elif self.cfgs.render_out_layer == 2:
                    render_feats_inuse_momentum = f_render_feats_momentum
                img_aggr_feat_momentum = self.momentum_phase6_aggregator(img_feats_inuse_momentum)
                render_aggr_feat_momentum = self.momentum_phase6_aggregator(render_feats_inuse_momentum)
                data_output['key_embeddings2'] = render_aggr_feat_momentum
                data_output['key_embeddings1'] = img_aggr_feat_momentum

        elif self.cfgs.phase == 7:
            render_aggr_mlp_feat = self.phase7_aggregator_mlp(self.phase7_mlp(render_feats_inuse))
            img_aggr_mlp_feat = self.phase7_aggregator_mlp(self.phase7_mlp(img_feats_inuse))

            with torch.no_grad(): # nmf is not derivable
                features = render_feats_inuse.contiguous()
                b, h, w = features.size(0), features.size(2), features.size(3)
                features = self.phase7_relu(features)
                flat_features = features.permute(0, 2, 3, 1).contiguous().view(-1, features.size(1))
                W, _ = NMF(flat_features, self.cfgs.phase7_K, random_seed=1, cuda=True, max_iter=self.cfgs.phase7_max_iter, verbose=False)
                isnan = torch.sum(torch.isnan(W).float())
                while isnan > 0:
                    print('nan detected. trying to resolve the nmf.')
                    W, _ = NMF(flat_features, self.cfgs.phase7_K, random_seed=random.randint(0, 255), cuda=True, max_iter=self.cfgs.phase7_max_iter, verbose=False)
                    isnan = torch.sum(torch.isnan(W).float())
                heatmaps = W.view(b, h, w, self.cfgs.phase7_K).permute(0,3,1,2)
                heatmaps = F.normalize(heatmaps, p=2, dim=1)
                heatmaps.requires_grad = False
                render_aggr_feat_NMF = self.phase7_aggregator_NMF(heatmaps)
                
            
            with torch.no_grad(): # nmf is not derivable
                features = img_feats_inuse.contiguous()
                b, h, w = features.size(0), features.size(2), features.size(3)
                features = self.phase7_relu(features)
                flat_features = features.permute(0, 2, 3, 1).contiguous().view(-1, features.size(1))
                W, _ = NMF(flat_features, self.cfgs.phase7_K, random_seed=1, cuda=True, max_iter=self.cfgs.phase7_max_iter, verbose=False)
                isnan = torch.sum(torch.isnan(W).float())
                while isnan > 0:
                    print('nan detected. trying to resolve the nmf.')
                    W, _ = NMF(flat_features, self.cfgs.phase7_K, random_seed=random.randint(0, 255), cuda=True, max_iter=self.cfgs.phase7_max_iter, verbose=False)
                    isnan = torch.sum(torch.isnan(W).float())
                heatmaps = W.view(b, h, w, self.cfgs.phase7_K).permute(0,3,1,2)
                heatmaps = F.normalize(heatmaps, p=2, dim=1)
                heatmaps.requires_grad = False
                img_aggr_feat_NMF = self.phase7_aggregator_NMF(heatmaps)
            

            render_aggr_feat_normal = self.phase7_aggregator_normal(render_feats_inuse)
            img_aggr_feat_normal = self.phase7_aggregator_normal(img_feats_inuse)
            img_aggr_feat_to_fuse = torch.cat([img_aggr_feat_normal, img_aggr_feat_NMF, img_aggr_mlp_feat], dim=1)
            render_aggr_feat_to_fuse = torch.cat([render_aggr_feat_normal, render_aggr_feat_NMF, render_aggr_mlp_feat], dim=1)
            img_aggr_fused_feat = self.phase7_fuse(img_aggr_feat_to_fuse)
            img_aggr_fused_feat = img_aggr_fused_feat.type(torch.float32)
            render_aggr_fused_feat = self.phase7_fuse(render_aggr_feat_to_fuse)
            render_aggr_fused_feat = render_aggr_fused_feat.type(torch.float32)
            data_output['embeddings1'] = img_aggr_fused_feat
            data_output['embeddings2'] = render_aggr_fused_feat
        elif self.cfgs.phase == 10:
            pass
        return data_output