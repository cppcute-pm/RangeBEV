import torch
import torch.nn as nn
from torch_scatter.composite import scatter_softmax
import torch_scatter
from timm.models.layers import DropPath
import einops
import torch.nn.functional as F
from pykeops.torch import LazyTensor


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class GroupedVectorSA(nn.Module):
    def __init__(self,
                 embed_channels,
                 cfgs,
                 ):
        super(GroupedVectorSA, self).__init__()
        self.embed_channels = embed_channels
        self.groups = cfgs.groups
        assert self.embed_channels % self.groups == 0
        self.attn_drop_rate = cfgs.attn_drop_rate
        self.qkv_bias = cfgs.qkv_bias
        self.pe_multiplier = cfgs.pe_multiplier
        self.pe_bias = cfgs.pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(self.embed_channels, self.embed_channels, bias=cfgs.qkv_bias),
            PointBatchNorm(self.embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(self.embed_channels, self.embed_channels, bias=cfgs.qkv_bias),
            PointBatchNorm(self.embed_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_v = nn.Linear(self.embed_channels, self.embed_channels, bias=cfgs.qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, self.embed_channels),
                PointBatchNorm(self.embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_channels, self.embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, self.embed_channels),
                PointBatchNorm(self.embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_channels, self.embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(self.embed_channels, self.groups),
            PointBatchNorm(self.groups),
            nn.ReLU(inplace=True),
            nn.Linear(self.groups, self.groups)
        )
        self.attn_drop = nn.Dropout(cfgs.attn_drop_rate)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, feats, coords, index):
        # assume the feats is (B, N, C)
        # assume the coords is (B, N, 3)
        # assume the index is (B, N, num_neighbors)

        B, N, C = feats.shape
        num_neighbors = index.shape[2]
        query, key, value = self.linear_q(feats), self.linear_k(feats), self.linear_v(feats) # (B, N, C)
        index_inuse = index.reshape(B, -1) # (B, N * num_neighbors)
        key = torch.gather(key, 
                           dim=1, 
                           index=index_inuse.unsqueeze(-1).expand(-1, -1, C)) # (B, N * num_neighbors, C)
        key = key.reshape(B, N, num_neighbors, C) # (B, N, num_neighbors, C)
        value = torch.gather(value, 
                             dim=1, 
                             index=index_inuse.unsqueeze(-1).expand(-1, -1, C))
        value = value.reshape(B, N, num_neighbors, C)
        coords_for_key = torch.gather(coords, 
                                      dim=1, 
                                      index=index_inuse.unsqueeze(-1).expand(-1, -1, 3))
        coords_for_key = coords_for_key.reshape(B, N, num_neighbors, 3)
        relation_qk = key - query.unsqueeze(-2) # (B, N, num_neighbors, C)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier((coords.unsqueeze(-2) - coords_for_key).reshape(B, -1, 3)) # (B, N * num_neighbors, C)
            relation_qk = relation_qk * pem.reshape(B, N, num_neighbors, C)
        if self.pe_bias:
            peb = self.linear_p_bias((coords.unsqueeze(-2) - coords_for_key).reshape(B, -1, 3)) # (B, N * num_neighbors, C)
            relation_qk = relation_qk + peb.reshape(B, N, num_neighbors, C)
            value = value + peb.reshape(B, N, num_neighbors, C)
        weight = self.weight_encoding(relation_qk.reshape(B, -1, C)) # (B, N * num_neighbors, self.groups)
        weight = self.softmax(weight.reshape(B, N, num_neighbors, self.groups)) # (B, N, num_neighbors, self.groups)
        weight = self.attn_drop(weight)
        value = einops.rearrange(value, "b n s (g i) -> b n s g i", g=self.groups)
        feats = torch.einsum("b n s g i, b n s g -> b n g i", value, weight)
        feats = einops.rearrange(feats, "b n g i ->b n (g i)")
        return feats


# class GroupedVectorCA(nn.Module):
#     def __init__(self,
#                  embed_channels,
#                  cfgs,
#                  ):
#         super(GroupedVectorCA, self).__init__()
#         self.embed_channels = embed_channels
#         self.groups = cfgs.groups
#         assert self.embed_channels % cfgs.groups == 0
#         self.attn_drop_rate = cfgs.attn_drop_rate
#         self.qkv_bias = cfgs.qkv_bias
#         self.pe_multiplier = cfgs.pe_multiplier
#         self.pe_bias = cfgs.pe_bias

#         self.linear_q = nn.Sequential(
#             nn.Linear(self.embed_channels, self.embed_channels, bias=cfgs.qkv_bias),
#             PointBatchNorm(self.embed_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.linear_k = nn.Sequential(
#             nn.Linear(self.embed_channels, self.embed_channels, bias=cfgs.qkv_bias),
#             PointBatchNorm(self.embed_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.linear_v = nn.Linear(self.embed_channels, self.embed_channels, bias=cfgs.qkv_bias)

#         if self.pe_multiplier:
#             self.linear_p_multiplier = nn.Sequential(
#                 nn.Linear(3, self.embed_channels),
#                 PointBatchNorm(self.embed_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(self.embed_channels, self.embed_channels),
#             )
#         if self.pe_bias:
#             self.linear_p_bias = nn.Sequential(
#                 nn.Linear(3, self.embed_channels),
#                 PointBatchNorm(self.embed_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(self.embed_channels, self.embed_channels),
#             )
#         self.weight_encoding = nn.Sequential(
#             nn.Linear(self.embed_channels, self.groups),
#             PointBatchNorm(self.groups),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.groups, self.groups)
#         )
#         self.attn_drop = nn.Dropout(cfgs.attn_drop_rate)

#     def forward(self, feats1, coords1, feats2, coords2, index):
#         # assume the feats1 is (B, N, C)
#         # assume the coords1 is (B, N, 3)
#         # assume the feats2 is (B, M+1, num_neighbors, C)
#         # assume the coords2 is (B, M+1, num_neighbors, 3)
#         # assume the index is (B, N)

#         B, N, C = feats1.shape
#         M = feats2.shape[1] - 1
#         num_neighbors = feats2.shape[2]
#         feats2 = feats2.reshape(B, (M + 1) * num_neighbors, C) # (B, (M+1) * num_neighbors, C)
#         query1, key1, value1 = self.linear_q(feats1), self.linear_k(feats1), self.linear_v(feats1) # (B, N, C)
#         query2, key2, value2 = self.linear_q(feats2), self.linear_k(feats2), self.linear_v(feats2) # (B, (M+1) * num_neighbors, C)
#         index_inuse = index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_neighbors, C) # (B, N, num_neighbors, C)
#         key2_in_1 = torch.gather(input=key2.reshape(B, (M+1), num_neighbors, C),
#                                  dim=1,
#                                  index=index_inuse) # (B, N, num_neighbors, C)
#         relation_qk_1 = key2_in_1 - query1.unsqueeze(-2) # (B, N, num_neighbors, C)
#         relation_qk_1 = relation_qk_1.reshape(B, -1, C) # (B, N * num_neighbors, C)
#         query2_in_1 = torch.gather(input=query2.reshape(B, (M+1), num_neighbors, C),
#                                    dim=1,
#                                    index=index_inuse) # (B, N, num_neighbors, C)
#         relation_qk_2 = key1.unsqueeze(-2) - query2_in_1 # (B, N, num_neighbors, C)
#         relation_qk_2 = relation_qk_2.reshape(B, -1, C) # (B, N * num_neighbors, C)
#         coords2_in_1 = torch.gather(input=coords2,
#                                     dim=1,
#                                     index=index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_neighbors, 3)) # (B, N, num_neighbors, 3)
#         value1 = value1.unsqueeze(-2).expand(-1, -1, num_neighbors, -1).reshape(B, -1, C) # (B, N * num_neighbors, C)
#         value2_in_1 = torch.gather(input=value2.reshape(B, (M+1), num_neighbors, C),
#                                     dim=1,
#                                     index=index_inuse) # (B, N, num_neighbors, C)
#         value2_in_1 = value2_in_1.reshape(B, -1, C) # (B, N * num_neighbors, C)
#         if self.pe_multiplier:
#             pem1 = self.linear_p_multiplier((coords1.unsqueeze(-2) - coords2_in_1).reshape(B, -1, C)) # (B, N * num_neighbors, C)
#             relation_qk_1 = relation_qk_1 * pem1
#             pem2 = self.linear_p_multiplier((coords2_in_1 - coords1.unsqueeze(-2)).reshape(B, -1, C)) # (B, N * num_neighbors, C)
#             relation_qk_2 = relation_qk_2 * pem2
#         if self.pe_bias:
#             peb1 = self.linear_p_bias((coords1.unsqueeze(-2) - coords2_in_1).reshape(B, -1, C)) # (B, N * num_neighbors, C)
#             relation_qk_1 = relation_qk_1 + peb1 
#             value2_in_1 = value2_in_1 + peb1
#             peb2 = self.linear_p_bias((coords2_in_1 - coords1.unsqueeze(-2)).reshape(B, -1, C)) # (B, N * num_neighbors, C)
#             relation_qk_2 = relation_qk_2 + peb2
#             value1 = value1 + peb2
#         weight_1 = self.weight_encoding(relation_qk_1) # (B, N * num_neighbors, self.groups)
#         weight_2 = self.weight_encoding(relation_qk_2) # (B, N * num_neighbors, self.groups)
#         weight_to_multiply_1 = F.softmax(weight_1.reshape(B, N, num_neighbors, self.groups),
#                                          dim=2) # (B, N, num_neighbors, self.groups)
#         weight_to_multiply_2 = scatter_softmax(src=weight_2.reshape(B, N, num_neighbors, self.groups),
#                                                 index=index.unsqueeze(-1).expand(-1, -1, -1, self.groups),
#                                                 dim=1,
#                                                 dim_size=M+1) # (B, N, num_neighbors, self.groups)
#         weight_to_output_1 = scatter_softmax(src=weight_1.reshape(B, N, num_neighbors, self.groups),
#                                                 index=index.unsqueeze(-1).expand(-1, -1, -1, self.groups),
#                                                 dim=1,
#                                                 dim_size=M+1) # (B, N, num_neighbors, self.groups)
#         weight_to_output_2 = F.softmax(src=weight_2.reshape(B, N, num_neighbors, self.groups),
#                                         dim=2) # (B, N, num_neighbors, self.groups)
#         weight_to_output_1 = torch.mean(weight_to_output_1, dim=-1).permute(0, 2, 1) # (B, num_neighbors, N)
#         weight_to_output_2 = torch.mean(weight_to_output_2, dim=-1).permute(0, 2, 1) # (B, num_neighbors, N)
#         weight_1 = self.attn_drop(weight_to_multiply_1) # (B, N * num_neighbors, self.groups)
#         weight_2 = self.attn_drop(weight_to_multiply_2) # (B, N * num_neighbors, self.groups)

#         mask_1 = torch.sign(index != M).float() # (B, N)
#         mask_1 = mask_1.unsqueeze(-1).expand(-1, -1, num_neighbors) # (B, N, num_neighbors)
#         mask_2 = torch.ones_like(coords2[..., 0])
#         mask_2[:, -1, :] = 0.0
#         mask_2_in_1 = torch.gather(input=mask_2,
#                               dim=1,
#                               index=index.unsqueeze(-1).expand(-1, -1, num_neighbors)) # (B, N, num_neighbors)
#         weight_1 = weight_1.reshape(B, N, num_neighbors, C) * mask_1.unsqueeze(-1)
#         weight_2 = weight_2.reshape(B, N, num_neighbors, C) * mask_2_in_1.unsqueeze(-1)
#         value2_in_1 = einops.rearrange(value2_in_1.reshape(B, N, num_neighbors, C), "b n s (g i) -> b n s g i", g=self.groups)
#         value1 = einops.rearrange(value1.reshape(B, N, num_neighbors, C), "b n s (g i) -> b n s g i", g=self.groups) # (B, N, num_neighbors, self.groups, C // self.groups)
#         feats1_output = torch.einsum("b n s g i, b n s g -> b n g i", value2_in_1, weight_1)
#         feats2_output = torch.mul(value1, weight_2.unsqueeze(-1)) # (B, N, num_neighbors, self.groups, C // self.groups)
#         feats2_output = torch_scatter.scatter_sum(feats2_output,
#                                                   index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_neighbors, self.groups, C // self.groups),
#                                                   dim=1,
#                                                   dim_size=M+1) # (B, M+1, num_neighbors, self.groups, C // self.groups)
#         feats1_output = einops.rearrange(feats1_output, "b n g i ->b n (g i)") # (B, N, C)
#         feats2_output = einops.rearrange(feats2_output, "b m s g i ->b m s (g i)") # (B, M+1, num_neighbors, C)
#         return feats1_output, weight_to_output_1, feats2_output, weight_to_output_2


# class CABlock(nn.Module):
#     def __init__(self,
#                  embed_channels,
#                  cfgs,
#                  ):
#         super(CABlock, self).__init__()
#         self.attn = GroupedVectorCA(embed_channels,
#                                     cfgs,
#                                     )
#         self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
#         self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
#         self.norm1 = PointBatchNorm(embed_channels)
#         self.norm2 = PointBatchNorm(embed_channels)
#         self.norm3 = PointBatchNorm(embed_channels)
#         self.act = nn.ReLU(inplace=True)
#         self.drop_path = DropPath(cfgs.drop_path_rate) if cfgs.drop_path_rate > 0. else nn.Identity()

#     def forward(self, feats1, coords1, feats2, coords2, index):
#         B, N, C = feats1.shape
#         _, M_plus, num_neighbors, _ = feats2.shape
#         identity1 = feats1
#         identity2 = feats2
#         feats1 = self.act(self.norm1(self.fc1(feats1)))
#         feats2 = self.act(self.norm1(self.fc1(feats2.reshape(B, -1, C))))
#         feats1, attn1, feats2, attn2 = self.attn(feats1, coords1, feats2.reshape(B, M_plus, num_neighbors, C), coords2, index)
#         feats1 = self.act(self.norm2(feats1))
#         feats2 = self.act(self.norm2(feats2.reshape(B, -1, C)))
#         feats1 = self.norm3(self.fc3(feats1))
#         feats1 = identity1 + self.drop_path(feats1)
#         feats1 = self.act(feats1)
#         feats2 = self.norm3(self.fc3(feats2))
#         feats2 = identity2 + self.drop_path(feats2).reshape(B, M_plus, num_neighbors, C)
#         feats2 = self.act(feats2)
#         return feats1, attn1, feats2, attn2


# class CABlockSeq(nn.Module):

#     def __init__(self, 
#                  cfgs, 
#                  embed_channels,
#                  depth):
#         super(CABlockSeq, self).__init__()
#         self.blocks = nn.ModuleList()
#         for _ in range(depth):
#             self.blocks.append(CABlock(embed_channels, cfgs))

#     def forward(self, feats1, coords1, feats2, coords2, index):
#         for block in self.blocks:
#             feats1, attn1, feats2, attn2 = block(feats1, coords1, feats2, coords2, index)
#         return feats1, attn1, feats2, attn2


class GroupedVectorCA(nn.Module):
    def __init__(self,
                 embed_channels,
                 cfgs,
                 ):
        super(GroupedVectorCA, self).__init__()
        self.embed_channels = embed_channels
        self.groups = cfgs.groups
        assert self.embed_channels % cfgs.groups == 0
        self.attn_drop_rate = cfgs.attn_drop_rate
        self.qkv_bias = cfgs.qkv_bias
        self.pe_multiplier = cfgs.pe_multiplier
        self.pe_bias = cfgs.pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(self.embed_channels, self.embed_channels, bias=cfgs.qkv_bias),
            PointBatchNorm(self.embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(self.embed_channels, self.embed_channels, bias=cfgs.qkv_bias),
            PointBatchNorm(self.embed_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_v = nn.Linear(self.embed_channels, self.embed_channels, bias=cfgs.qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, self.embed_channels),
                PointBatchNorm(self.embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_channels, self.embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, self.embed_channels),
                PointBatchNorm(self.embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_channels, self.embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(self.embed_channels, self.groups),
            PointBatchNorm(self.groups),
            nn.ReLU(inplace=True),
            nn.Linear(self.groups, self.groups)
        )
        self.attn_drop = nn.Dropout(cfgs.attn_drop_rate)

    def forward(self, feats1, coords1, feats2, coords2):
        # assume the feats1 is (B, N, C)
        # assume the coords1 is (B, N, 3)
        # assume the feats2 is (B, M, C)
        # assume the coords2 is (B, M, 3)

        B, N, C = feats1.shape
        M = feats2.shape[1]
        query1, key1, value1 = self.linear_q(feats1), self.linear_k(feats1), self.linear_v(feats1) # (B, N, C)
        query2, key2, value2 = self.linear_q(feats2), self.linear_k(feats2), self.linear_v(feats2) # (B, M, C)
        relation_qk_1 = key2.unsqueeze(1) - query1.unsqueeze(2) # (B, N, M, C)
        relation_qk_1 = relation_qk_1.reshape(B, -1, C) # (B, N * M, C)
        relation_qk_2 = key1.unsqueeze(1) - query2.unsqueeze(2) # (B, M, N, C)
        relation_qk_2 = relation_qk_2.reshape(B, -1, C) # (B, M * N, C)
        value1 = value1.unsqueeze(1).expand(-1, M, -1, -1).reshape(B, -1, C) # (B, M * N, C)
        value2 = value2.unsqueeze(1).expand(-1, N, -1, -1).reshape(B, -1, C) # (B, N * M, C)
        if self.pe_multiplier:
            pem1 = self.linear_p_multiplier((coords1.unsqueeze(-2) - coords2.unsqueeze(1)).reshape(B, -1, 3)) # (B, N * M, C)
            relation_qk_1 = relation_qk_1 * pem1
            pem2 = self.linear_p_multiplier((coords2.unsqueeze(-2) - coords1.unsqueeze(1)).reshape(B, -1, 3)) # (B, M * N, C)
            relation_qk_2 = relation_qk_2 * pem2
        if self.pe_bias:
            peb1 = self.linear_p_bias((coords1.unsqueeze(-2) - coords2.unsqueeze(1)).reshape(B, -1, 3)) # (B, N * M, C)
            relation_qk_1 = relation_qk_1 + peb1 
            value2 = value2 + peb1
            peb2 = self.linear_p_bias((coords2.unsqueeze(-2) - coords1.unsqueeze(1)).reshape(B, -1, 3)) # (B, M * N, C)
            relation_qk_2 = relation_qk_2 + peb2
            value1 = value1 + peb2
        weight_1 = self.weight_encoding(relation_qk_1) # (B, N * M, self.groups)
        weight_2 = self.weight_encoding(relation_qk_2) # (B, M * N, self.groups)
        weight_to_multiply_1 = F.softmax(weight_1.reshape(B, N, M, self.groups), dim=2) # (B, N, M, self.groups)
        weight_to_multiply_2 = F.softmax(weight_2.reshape(B, M, N, self.groups), dim=2) # (B, M, N, self.groups)
        weight_to_output_1 = F.softmax(weight_1.reshape(B, N, M, self.groups), dim=1) # (B, N, M, self.groups)
        weight_to_output_2 = F.softmax(weight_2.reshape(B, M, N, self.groups), dim=1) # (B, M, N, self.groups)
        weight_to_output_1 = torch.mean(weight_to_output_1, dim=-1) # (B, N, M)
        weight_to_output_2 = torch.mean(weight_to_output_2, dim=-1) # (B, M, N)
        weight_1 = self.attn_drop(weight_to_multiply_1) # (B, N, M, self.groups)
        weight_2 = self.attn_drop(weight_to_multiply_2) # (B, M, N, self.groups)
        value2 = einops.rearrange(value2.reshape(B, N, M, C), "b n m (g i) -> b n m g i", g=self.groups)
        value1 = einops.rearrange(value1.reshape(B, M, N, C), "b m n (g i) -> b m n g i", g=self.groups)
        feats1_output = torch.einsum("b n m g i, b n m g -> b n g i", value2, weight_1)
        feats2_output = torch.einsum("b m n g i, b m n g -> b m g i", value1, weight_2)
        feats1_output = einops.rearrange(feats1_output, "b n g i ->b n (g i)") # (B, N, C)
        feats2_output = einops.rearrange(feats2_output, "b m g i ->b m (g i)") # (B, M, C)
        return feats1_output, weight_to_output_1, feats2_output, weight_to_output_2


class CABlock(nn.Module):
    def __init__(self,
                 embed_channels,
                 cfgs,
                 ):
        super(CABlock, self).__init__()
        self.attn = GroupedVectorCA(embed_channels,
                                    cfgs,
                                    )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop_path = DropPath(cfgs.drop_path_rate) if cfgs.drop_path_rate > 0. else nn.Identity()

    def forward(self, feats1, coords1, feats2, coords2):
        identity1 = feats1
        identity2 = feats2
        feats1 = self.act(self.norm1(self.fc1(feats1)))
        feats2 = self.act(self.norm1(self.fc1(feats2)))
        feats1, attn1, feats2, attn2 = self.attn(feats1, coords1, feats2, coords2)
        feats1 = self.act(self.norm2(feats1))
        feats2 = self.act(self.norm2(feats2))
        feats1 = self.norm3(self.fc3(feats1))
        feats1 = identity1 + self.drop_path(feats1)
        feats1 = self.act(feats1)
        feats2 = self.norm3(self.fc3(feats2))
        feats2 = identity2 + self.drop_path(feats2)
        feats2 = self.act(feats2)
        return feats1, attn1, feats2, attn2


class CABlockSeq(nn.Module):

    def __init__(self, 
                 cfgs, 
                 embed_channels,
                 depth):
        super(CABlockSeq, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(CABlock(embed_channels, cfgs))

    def forward(self, feats1, coords1, feats2, coords2):
        for block in self.blocks:
            feats1, attn1, feats2, attn2 = block(feats1, coords1, feats2, coords2)
        return feats1, attn1, feats2, attn2


class SABlock(nn.Module):
    def __init__(self,
                 embed_channels,
                 cfgs,
                 ):
        super(SABlock, self).__init__()
        self.attn = GroupedVectorSA(embed_channels,
                                    cfgs,
                                    )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop_path = DropPath(cfgs.drop_path_rate) if cfgs.drop_path_rate > 0. else nn.Identity()

    def forward(self, feats, coords, index):
        identity = feats
        feats = self.act(self.norm1(self.fc1(feats)))
        feats = self.attn(feats, coords, index)
        feats = self.act(self.norm2(feats))
        feats = self.norm3(self.fc3(feats))
        feats = identity + self.drop_path(feats)
        feats = self.act(feats)
        return feats

class SABlockSeq(nn.Module):

    def __init__(self, 
                 cfgs, 
                 embed_channels,
                 depth):
        super(SABlockSeq, self).__init__()
        self.neighbours = cfgs.neighbours
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(SABlock(embed_channels, cfgs))

    def forward(self, feats, coords):
        
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        coords_lazy_1 = LazyTensor(coords.unsqueeze(-2)) # (B, N, -1, 3)
        coords_lazy_2 = LazyTensor(coords.unsqueeze(-3)) # (B, -1, N, 3)
        coords_to_coords_dist = (coords_lazy_1 - coords_lazy_2).norm2() # (B, N, N)
        device_id = feats.device.index
        _, index = coords_to_coords_dist.Kmin_argKmin(K=self.neighbours, dim=2, device_id=device_id)
        for block in self.blocks:
            feats = block(feats, coords, index)
        return feats