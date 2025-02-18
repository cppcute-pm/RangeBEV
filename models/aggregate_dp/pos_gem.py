import torch.nn as nn
import torch
import torch_scatter
import torch.nn.functional as F

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)

class PoS_GeM(nn.Module):
    def __init__(self, cfgs):
        super(PoS_GeM, self).__init__()
        p = cfgs.p
        eps = cfgs.eps
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        self.eps = eps
        self.type = cfgs.type
        if self.type == 'type_3':
            self.mlp_1 = nn.Sequential(
                nn.Conv1d(2 * 256, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_2 = nn.Sequential(
                nn.Conv1d(2 * 256, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_3 = nn.Sequential(
                nn.Conv1d(2 * 256, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_4 = nn.Sequential(
                nn.Conv1d(2 * 256, 256, 1),
                nn.BatchNorm1d(256),
            )
        if self.type == 'type_3_1':
            self.coord_dim_type = cfgs.coord_dim_type
            if self.coord_dim_type == 1:
                coord_dim = 3
            elif self.coord_dim_type == 2:
                coord_dim = 2
            self.mlp_1 = nn.Sequential(
                nn.Conv1d(2 * 256 + coord_dim, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_2 = nn.Sequential(
                nn.Conv1d(2 * 256 + coord_dim, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_3 = nn.Sequential(
                nn.Conv1d(2 * 256 + coord_dim, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_4 = nn.Sequential(
                nn.Conv1d(2 * 256 + coord_dim, 256, 1),
                nn.BatchNorm1d(256),
            )
        if self.type == 'type_5':
            self.coord_dim_type = cfgs.coord_dim_type
            if self.coord_dim_type == 1:
                coord_dim = 3
            elif self.coord_dim_type == 2:
                coord_dim = 2
            self.mlp_1_1 = nn.Sequential(
                nn.Conv1d(coord_dim, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_2_1 = nn.Sequential(
                nn.Conv1d(coord_dim, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_3_1 = nn.Sequential(
                nn.Conv1d(coord_dim, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_4_1 = nn.Sequential(
                nn.Conv1d(coord_dim, 256, 1),
                nn.BatchNorm1d(256),
            )

            self.mlp_1_2 = nn.Sequential(
                nn.Conv1d(2 * 256, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_2_2 = nn.Sequential(
                nn.Conv1d(2 * 256, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_3_2 = nn.Sequential(
                nn.Conv1d(2 * 256, 256, 1),
                nn.BatchNorm1d(256),
            )
            self.mlp_4_2 = nn.Sequential(
                nn.Conv1d(2 * 256, 256, 1),
                nn.BatchNorm1d(256),
            )


    def forward(self, x, index, coords):
        """
        args:
            x: shape (B, C, N)
            index: shape [(B, N), (B, N1), (B, N2), (B, N3), ..., (B, 1)]
            coords: shape [(B, N, 3), (B, N1, 3), (B, N2, 3), (B, N3, 3), ..., (B, 1, 3)]
        """
        assert len(index) == len(coords) == len(self.p) == len(self.eps)
        if self.type == 'type_1':
            curr_x = x
            for i in range(len(index) - 1):
                B, curr_C, curr_N = curr_x.shape
                curr_coords_dim = coords[i].shape[-1]
                coords_next_in_curr = torch.gather(input=coords[i+1],
                            dim=1,
                            index=index[i].unsqueeze(-1).expand(-1, -1, curr_coords_dim)) # (B, N, 3)
                coords_curr_delta_next = coords[i] - coords_next_in_curr # (B, N, 3)
                coords_curr_delta_next_max, _ = torch_scatter.scatter_max(coords_curr_delta_next, 
                                                                index[i].unsqueeze(-1).expand(-1, -1, curr_coords_dim),
                                                                dim=1,
                                                                dim_size=index[i+1].shape[1]) # (B, N1, 3)
                coords_curr_delta_next_min, _ = torch_scatter.scatter_min(coords_curr_delta_next, 
                                                                index[i].unsqueeze(-1).expand(-1, -1, curr_coords_dim),
                                                                dim=1,
                                                                dim_size=index[i+1].shape[1]) # (B, N1, 3)
                pos_upper_bound = torch.clamp(coords_curr_delta_next_max, min=0.0) # (B, N1, 3)
                neg_lower_bound = torch.clamp(coords_curr_delta_next_min, max=0.0) # (B, N1, 3)
                pos_upper_bound_next_in_curr = torch.gather(pos_upper_bound,
                                                            dim=1,
                                                            index=index[i].unsqueeze(-1).expand(-1, -1, curr_coords_dim))
                neg_lower_bound_next_in_curr = torch.gather(neg_lower_bound,
                                                            dim=1,
                                                            index=index[i].unsqueeze(-1).expand(-1, -1, curr_coords_dim))
                coords_curr_delta_next_mask = torch.ge(coords_curr_delta_next, 0.0) # (B, N, 3)
                inuse_upper_bound = torch.where(coords_curr_delta_next_mask, pos_upper_bound_next_in_curr, neg_lower_bound_next_in_curr) # (B, N, 3)
                weight_curr_to_next_reverse = torch.div(torch.abs(coords_curr_delta_next), torch.abs(inuse_upper_bound) + 1e-6) # (B, N, 3)
                weight_curr_to_next = 1.0 - weight_curr_to_next_reverse # (B, N, 3)
                x_pow = curr_x.clamp(min=self.eps[i]).pow(self.p[i]) # (B, C, N)
                remainder = curr_C % curr_coords_dim
                quotient = curr_C // curr_coords_dim
                inuse_weight_curr_to_next = []
                for j in range(curr_coords_dim):
                    if j == (curr_coords_dim - 1):
                        inuse_weight_curr_to_next.append(weight_curr_to_next[..., j:j+1].expand(-1, -1, quotient + remainder))
                    else:
                        inuse_weight_curr_to_next.append(weight_curr_to_next[..., j:j+1].expand(-1, -1, quotient))
                inuse_weight_curr_to_next = torch.cat(inuse_weight_curr_to_next, dim=-1) # (B, N, C)
                inuse_weight_curr_to_next = inuse_weight_curr_to_next.permute(0, 2, 1) # (B, C, N)
                x_pow_weighted = x_pow * inuse_weight_curr_to_next # (B, C, N)
                x_pow_weighted_sum = torch_scatter.scatter_sum(x_pow_weighted, 
                                                            index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                            dim=-1,
                                                            dim_size=index[i+1].shape[1]) # (B, C, N1)
                inuse_weight_curr_to_next_sum = torch_scatter.scatter_sum(inuse_weight_curr_to_next,
                                                                    index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                                    dim=-1,
                                                                    dim_size=index[i+1].shape[1]) # (B, C, N1)
                x_pow_weighted_sum = x_pow_weighted_sum / (inuse_weight_curr_to_next_sum + 1e-6) # (B, C, N1)
                next_x = x_pow_weighted_sum.pow(1./self.p[i]) # (B, C, N1)
                curr_x = next_x
            x_out = curr_x.squeeze(-1) # (B, C)
            return x_out

        elif self.type == 'type_2':
            curr_x = x
            for i in range(len(index) - 1):
                B, curr_C, curr_N = curr_x.shape
                curr_coords_dim = coords[i].shape[-1]
                x_pow = curr_x.clamp(min=self.eps[i]).pow(self.p[i]) # (B, C, N)
                x_pow_sum = torch_scatter.scatter_sum(x_pow, 
                                                    index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                    dim=-1,
                                                    dim_size=index[i+1].shape[1]) # (B, C, N1)
                inuse_weight_curr_to_next_sum = torch_scatter.scatter_sum(torch.ones_like(x_pow),
                                                                    index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                                    dim=-1,
                                                                    dim_size=index[i+1].shape[1]) # (B, C, N1)
                x_pow_sum = x_pow_sum / (inuse_weight_curr_to_next_sum + 1e-6) # (B, C, N1)
                next_x = x_pow_sum.pow(1./self.p[i]) # (B, C, N1)
                curr_x = next_x
            x_out = curr_x.squeeze(-1) # (B, C)
            return x_out

        elif self.type == 'type_3':
            curr_x = x
            for i in range(len(index) - 1):
                B, curr_C, curr_N = curr_x.shape
                curr_coords_dim = coords[i].shape[-1]
                coords_next_in_curr = torch.gather(input=coords[i+1],
                                                    dim=1,
                                                    index=index[i].unsqueeze(-1).expand(-1, -1, curr_coords_dim)) # (B, N, 3)
                coords_curr_delta_next = coords[i] - coords_next_in_curr # (B, N, 3)
                dist_curr_to_next = torch.cdist(coords[i], coords[i+1], p=2.0) # (B, N, N1)
                index_next_in_curr = torch.argmin(dist_curr_to_next, dim=1, keepdim=False) # (B, N1)
                x_next_in_curr = torch.gather(input=curr_x, 
                                            dim=2,
                                            index=index_next_in_curr.unsqueeze(1).expand(-1, curr_C, -1)) # (B, C, N1)
                x_next_in_curr = torch.gather(input=x_next_in_curr,
                                            dim=2,
                                            index=index[i].unsqueeze(1).expand(-1, curr_C, -1)) # (B, C, N)
                x_next_delta_next = x_next_in_curr - curr_x # (B, C, N)
                middle_x = torch.cat((x_next_in_curr, x_next_delta_next), dim=1) # (B, 2C, N)
                curr_net = getattr(self, 'mlp_{}'.format(i+1))
                middle_x = curr_net(middle_x)
                middle_x = middle_x + curr_x
                x_pow = middle_x.clamp(min=self.eps[i]).pow(self.p[i]) # (B, C, N)
                x_pow_sum = torch_scatter.scatter_sum(x_pow, 
                                                    index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                    dim=-1,
                                                    dim_size=index[i+1].shape[1]) # (B, C, N1)
                inuse_weight_curr_to_next_sum = torch_scatter.scatter_sum(torch.ones_like(x_pow),
                                                                    index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                                    dim=-1,
                                                                    dim_size=index[i+1].shape[1]) # (B, C, N1)
                x_pow_sum = x_pow_sum / (inuse_weight_curr_to_next_sum + 1e-6) # (B, C, N1)
                next_x = x_pow_sum.pow(1./self.p[i]) # (B, C, N1)
                curr_x = next_x
            x_out = curr_x.squeeze(-1) # (B, C)
            return x_out
        
        elif self.type == 'type_3_1':
            curr_x = x
            for i in range(len(index) - 1):
                B, curr_C, curr_N = curr_x.shape
                curr_coords_dim = coords[i].shape[-1]
                coords_next_in_curr = torch.gather(input=coords[i+1],
                                                    dim=1,
                                                    index=index[i].unsqueeze(-1).expand(-1, -1, curr_coords_dim)) # (B, N, 3)
                coords_curr_delta_next = coords[i] - coords_next_in_curr # (B, N, 3)
                dist_curr_to_next = torch.cdist(coords[i], coords[i+1], p=2.0) # (B, N, N1)
                index_next_in_curr = torch.argmin(dist_curr_to_next, dim=1, keepdim=False) # (B, N1)
                x_next_in_curr = torch.gather(input=curr_x, 
                                            dim=2,
                                            index=index_next_in_curr.unsqueeze(1).expand(-1, curr_C, -1)) # (B, C, N1)
                x_next_in_curr = torch.gather(input=x_next_in_curr,
                                            dim=2,
                                            index=index[i].unsqueeze(1).expand(-1, curr_C, -1)) # (B, C, N)
                x_next_delta_next = x_next_in_curr - curr_x # (B, C, N)
                middle_x = torch.cat((x_next_in_curr, x_next_delta_next, coords_curr_delta_next.permute(0, 2, 1)), dim=1)
                curr_net = getattr(self, 'mlp_{}'.format(i+1))
                middle_x = curr_net(middle_x)
                middle_x = middle_x + curr_x
                x_pow = middle_x.clamp(min=self.eps[i]).pow(self.p[i]) # (B, C, N)
                x_pow_sum = torch_scatter.scatter_sum(x_pow, 
                                                    index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                    dim=-1,
                                                    dim_size=index[i+1].shape[1]) # (B, C, N1)
                inuse_weight_curr_to_next_sum = torch_scatter.scatter_sum(torch.ones_like(x_pow),
                                                                    index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                                    dim=-1,
                                                                    dim_size=index[i+1].shape[1]) # (B, C, N1)
                x_pow_sum = x_pow_sum / (inuse_weight_curr_to_next_sum + 1e-6) # (B, C, N1)
                next_x = x_pow_sum.pow(1./self.p[i]) # (B, C, N1)
                curr_x = next_x
            x_out = curr_x.squeeze(-1) # (B, C)
            return x_out

        elif self.type == 'type_4':
            curr_x = x
            for i in range(len(index) - 1):
                B, curr_C, curr_N = curr_x.shape
                curr_coords_dim = coords[i].shape[-1]
                coords_next_in_curr = torch.gather(input=coords[i+1],
                            dim=1,
                            index=index[i].unsqueeze(-1).expand(-1, -1, curr_coords_dim)) # (B, N, 3)
                coords_curr_delta_next = coords[i] - coords_next_in_curr # (B, N, 3)
                radius_curr_delta_next = torch.linalg.norm(coords_curr_delta_next, dim=-1, keepdim=False) # (B, N)
                radius_curr_delta_next_max, _ = torch_scatter.scatter_max(radius_curr_delta_next, 
                                                                index[i],
                                                                dim=1,
                                                                dim_size=index[i+1].shape[1]) # (B, N1)
                radius_curr_delta_next_max = torch.gather(radius_curr_delta_next_max,
                                                        dim=1,
                                                        index=index[i]) # (B, N)

                weight_curr_to_next_reverse = torch.div(torch.abs(radius_curr_delta_next), torch.abs(radius_curr_delta_next_max) + 1e-6) # (B, N)
                weight_curr_to_next = 1.0 - weight_curr_to_next_reverse
                inuse_weight_curr_to_next = weight_curr_to_next.unsqueeze(1).expand(-1, curr_C, -1) # (B, C, N)
                x_pow = curr_x.clamp(min=self.eps[i]).pow(self.p[i]) # (B, C, N)
                x_pow_weighted = x_pow * inuse_weight_curr_to_next # (B, C, N)
                x_pow_weighted_sum = torch_scatter.scatter_sum(x_pow_weighted, 
                                                            index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                            dim=-1,
                                                            dim_size=index[i+1].shape[1]) # (B, C, N1)
                inuse_weight_curr_to_next_sum = torch_scatter.scatter_sum(inuse_weight_curr_to_next,
                                                                    index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                                    dim=-1,
                                                                    dim_size=index[i+1].shape[1]) # (B, C, N1)
                x_pow_weighted_sum = x_pow_weighted_sum / (inuse_weight_curr_to_next_sum + 1e-6) # (B, C, N1)
                next_x = x_pow_weighted_sum.pow(1./self.p[i]) # (B, C, N1)
                # if self.normalize:
                #     curr_x = F.normalize(next_x, p=2.0, dim=1)
                # else:
                curr_x = next_x
            x_out = curr_x.squeeze(-1) # (B, C)
            return x_out
        
        elif self.type == 'type_5': # 将3_1改进成位置编码的形式
            curr_x = x
            for i in range(len(index) - 1):
                B, curr_C, curr_N = curr_x.shape
                curr_coords_dim = coords[i].shape[-1]
                coords_next_in_curr = torch.gather(input=coords[i+1],
                                                    dim=1,
                                                    index=index[i].unsqueeze(-1).expand(-1, -1, curr_coords_dim)) # (B, N, 3)
                coords_curr_delta_next = coords[i] - coords_next_in_curr # (B, N, 3)
                dist_curr_to_next = torch.cdist(coords[i], coords[i+1], p=2.0) # (B, N, N1)
                index_next_in_curr = torch.argmin(dist_curr_to_next, dim=1, keepdim=False) # (B, N1)
                x_next_in_curr = torch.gather(input=curr_x, 
                                            dim=2,
                                            index=index_next_in_curr.unsqueeze(1).expand(-1, curr_C, -1)) # (B, C, N1)
                x_next_in_curr = torch.gather(input=x_next_in_curr,
                                            dim=2,
                                            index=index[i].unsqueeze(1).expand(-1, curr_C, -1)) # (B, C, N)
                x_next_delta_next = x_next_in_curr - curr_x # (B, C, N)
                middle_net_1 = getattr(self, 'mlp_{}_1'.format(i+1))
                middle_feats_1 = middle_net_1(coords_curr_delta_next.permute(0, 2, 1))  # (B, C, N)
                middle_feats_2 = middle_feats_1 + x_next_delta_next
                middle_net_2 = getattr(self, 'mlp_{}_2'.format(i+1))
                middle_feats_3 = torch.cat((x_next_in_curr, middle_feats_2), dim=1)
                middle_x = middle_net_2(middle_feats_3)
                middle_x = middle_x + curr_x
                x_pow = middle_x.clamp(min=self.eps[i]).pow(self.p[i]) # (B, C, N)
                x_pow_sum = torch_scatter.scatter_sum(x_pow, 
                                                    index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                    dim=-1,
                                                    dim_size=index[i+1].shape[1]) # (B, C, N1)
                inuse_weight_curr_to_next_sum = torch_scatter.scatter_sum(torch.ones_like(x_pow),
                                                                    index[i].unsqueeze(1).expand(-1, curr_C, -1), # (B, C, N)
                                                                    dim=-1,
                                                                    dim_size=index[i+1].shape[1]) # (B, C, N1)
                x_pow_sum = x_pow_sum / (inuse_weight_curr_to_next_sum + 1e-6) # (B, C, N1)
                next_x = x_pow_sum.pow(1./self.p[i]) # (B, C, N1)
                curr_x = next_x
            x_out = curr_x.squeeze(-1) # (B, C)
            return x_out